import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import resnet50
from torchvision.models.detection import keypointrcnn_resnet50_fpn

def weights_init_kaiming(layer):
    if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
        init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(layer, nn.Linear):
        init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            init.constant_(layer.bias, 0.0)
    elif isinstance(layer, nn.BatchNorm1d):
        init.normal_(layer.weight, mean=1.0, std=0.02)
        init.constant_(layer.bias, 0.0)

def weights_init_classifier(layer):
    if isinstance(layer, nn.Linear):
        init.normal_(layer.weight, std=0.001)
        init.constant_(layer.bias, 0.0)

class SGConv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.phi = nn.Linear(dim_in, dim_in, bias=False)
        self.psi = nn.Linear(dim_in, dim_in, bias=False)
        self.W0  = nn.Linear(dim_in, dim_out, bias=False)
        self.W1  = nn.Linear(dim_in, dim_out, bias=False)
        self.act = nn.ReLU()
        for l in (self.phi, self.psi, self.W0, self.W1):
            weights_init_kaiming(l)

    def forward(self, X, overlap_A):
        scores = self.phi(X) @ self.psi(X).T
        S = F.softmax(scores, dim=1)
        S = S.masked_fill(S < 0.01, 0)
        A = overlap_A + S
        I     = torch.eye(A.size(0), device=A.device)
        selfA = I * A
        nbrA  = (1 - I) * A
        m0 = self.W0((X.T @ selfA).T)
        m1 = self.W1((X.T @ nbrA).T)
        return self.act(m0 + m1)

class PCB(nn.Module):
    def __init__(self, num_classes, knn_k: int = 10):
        super().__init__()
        self.num_parts = 9
        self.knn_k     = knn_k

        backbone = resnet50(pretrained=True)
        backbone.fc = nn.Identity()
        backbone.layer4[0].downsample[0].stride = (1, 1)
        backbone.layer4[0].conv2.stride      = (1, 1)
        self.feature_extractor = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

        self.pose_model = keypointrcnn_resnet50_fpn(pretrained=True).eval()
        for p in self.pose_model.parameters():
            p.requires_grad = False

        self.reduce_conv = nn.Conv1d(2048, 512, kernel_size=1, bias=False)
        init.kaiming_normal_(self.reduce_conv.weight, mode='fan_in', nonlinearity='relu')

        dims = [512, 512, 256, 256, 256]
        self.sgconvs = nn.ModuleList([
            SGConv(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])

        for m in range(self.num_parts):
            clf = nn.Sequential(
                nn.Linear(256, num_classes),
                nn.BatchNorm1d(num_classes)
            )
            clf.apply(weights_init_classifier)
            setattr(self, f'classifier{m}', clf)

        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x, training: bool = False):
        B, C, H, W = x.shape

        feats = self.feature_extractor(x)
        _, Cmap, Hmap, _ = feats.shape

        mean = x.new_tensor([0.485,0.456,0.406]).view(1,3,1,1)
        std  = x.new_tensor([0.229,0.224,0.225]).view(1,3,1,1)
        denorm = x * std + mean
        with torch.no_grad():
            dets = self.pose_model([img for img in denorm])

        R = feats.new_zeros(B, Cmap, self.num_parts)
        for i in range(B):
            det = dets[i]
            if len(det['keypoints']) == 0:
                R[i] = F.adaptive_max_pool2d(feats[i], (self.num_parts,1)).squeeze(-1)
            else:
                idx = det['scores'].argmax()
                kpts = det['keypoints'][idx][:,:2]
                ys   = kpts[:,1] / H * Hmap
                qs   = torch.linspace(0,1,self.num_parts+1,device=ys.device)
                bnds = torch.quantile(ys, qs).long().clamp(0, Hmap-1)
                for m in range(self.num_parts):
                    y1,y2 = bnds[m].item(), bnds[m+1].item()
                    if y2 <= y1: y2 = min(y1+1, Hmap)
                    region = feats[i,:,y1:y2,:].reshape(Cmap, -1)
                    R[i,:,m], _ = region.max(dim=1)

        P = self.reduce_conv(R)

        flat = P.permute(0,2,1).reshape(B, -1)
        dist = torch.cdist(flat, flat, p=2)
        k = min(self.knn_k + 1, B)
        knn = dist.topk(k, largest=False).indices
        # drop self index if possible
        if k > 1:
            knn = knn[:, 1:]
        O = torch.zeros(B, B, device=x.device)
        for i in range(B):
            si = set(knn[i].tolist())
            for j in range(B):
                O[i,j] = len(si & set(knn[j].tolist()))

        deg = O.sum(1)
        inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        D_inv = torch.diag(inv_sqrt)
        O_norm = D_inv @ O @ D_inv
        O_norm.fill_diagonal_(1.0)

        Q_parts = []
        for m in range(self.num_parts):
            Xi = P[:,:,m]
            for sg in self.sgconvs:
                Xi = sg(Xi, O_norm)
            Q_parts.append(Xi)

        Q = torch.stack(Q_parts, dim=2)

        if training:
            logits = []
            dropped = self.dropout(Q)
            for m in range(self.num_parts):
                feat = dropped[:,:,m]
                logits.append(getattr(self, f'classifier{m}')(feat))
            return logits
        else:
            return Q
