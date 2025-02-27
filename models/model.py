
from copy import deepcopy
import torch
import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from utils import printProgressBar
from modules.partialconv2d import PartialConv2d as PConv2D
from modules.submodules import CRPSLoss, EarlyStopping
from modules.tft_submodules import VariableSelectionNetwork


class TelNet(nn.Module):

    def __init__(self, N, H, W, I, D, T, L, drop, weight_scale):
        super(TelNet, self).__init__()
        self.L = L

        self.auto_emb = nn.Sequential(
            PConv2D(1, D//4, kernel_size=(2, 2), stride=(2, 2), multi_channel=False, return_mask=True),
            PConv2D(D//4, D//2, kernel_size=(2, 2), stride=(2, 2), multi_channel=False, return_mask=True),
            PConv2D(D//2, D, kernel_size=(H//4, W//4), stride=(H//4, W//4), multi_channel=False, return_mask=False)
        )
        self.cov_emb = nn.ModuleList([nn.Linear(1, D, False) for _ in range(I)])
        self.static_emb = nn.Linear(12, D, False)
        self.varsel = VariableSelectionNetwork({f'{i}':D for i in range(I+1)}, D, 
                                               input_embedding_flags={f'{i}':True for i in range(I+1)}, 
                                               dropout=drop)
        self.layer_norm1 = nn.LayerNorm(D)
        self.decoder = nn.LSTM(D, D, 1, batch_first=True, dropout=drop)
        self.layer_norm2 = nn.LayerNorm(D)
        self.layer_norm3 = nn.LayerNorm(D)
        # self.attn = InterpretableMultiHeadAttention(1, D, drop)
        self.head = nn.Linear(D, N*H*W)
        if weight_scale != 0:
            self.init_weights(weight_scale)
    
    def init_weights(self, scale=0.02):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.trunc_normal_(param, std=scale)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=scale)
                nn.init.zeros_(m.bias)
        
    def forward(self, X, xmask):

        Xauto, Xcov, Xstatic = X
        B, T, H, W = Xauto.shape  # B, T, H, W
        _, _, I = Xcov.shape  # B, T, I

        xmask = xmask[0:1]
        Xstatic = torch.stack([torch.roll(Xstatic, i, 1) for i in range(-T+1, self.L+1)], 1)
        Xstatic = self.static_emb(Xstatic.view(B*(T+self.L), -1)).view(B, T+self.L, -1)
        Xstatic_lag = Xstatic[:, :T]  # B, T, D
        Xstatic_lead = Xstatic[:, T:]  # B, L, D
        # Xstatic_lead = self.lead_emb(Xstatic_lead.view(B*self.L, -1)).view(B, self.L, -1)  # B, L, D
        
        Xauto_emb = []
        for t in range(T):
            Xin = (Xauto[:, t:t+1], xmask)
            Xout = self.auto_emb(Xin)  # B, D, 1, 1
            Xauto_emb.append(Xout.squeeze((2, 3)))
        Xauto_emb = torch.stack(Xauto_emb, 1)  # B, T, D

        Xcov_emb = []
        for i in range(I):
            Xcov_emb_i = []
            for t in range(T):
                Xout = self.cov_emb[i](Xcov[:, t:t+1, i])
                Xcov_emb_i.append(Xout)
            Xcov_emb.append(torch.stack(Xcov_emb_i, 1))
        Xcov_emb = torch.stack(Xcov_emb, 2)  # B, T, I, D

        Xemb = torch.concat((Xauto_emb[:, :, None], Xcov_emb), 2)  # B, T, I+1, D

        Xsel = []
        for t in range(T):
            Xin = {f'{p}': Xemb[:, t, p] + Xstatic_lag[:, t] for p in range(Xemb.shape[2])}
            Xout, varsel_wgts = self.varsel(Xin)  # [B, D], [B, I+1]
            Xsel.append(Xout)
        Xemb = torch.stack(Xsel, 1)  # B, T, D
        Xemb = self.layer_norm1(Xemb)  # B, T, D

        yemb = []
        wgts = [varsel_wgts]
        for l in range(self.L):
            ydec, _ = self.decoder(Xemb)  # B, T, D  or B, 1, D
            ydec = self.layer_norm2(ydec[:, -1:])  # B, 1, D
            ydec = ydec + Xstatic_lead[:, l:l+1]  # B, 1, D
            yemb.append(ydec.squeeze(1))  # B, 1, D
            if l < self.L-1:
                Xemb_ = torch.concat((ydec[:, :, None], Xcov_emb[:, -1:]), 2)  # B, 1, I+1, D
                Xin = {f'{p}': Xemb_[:, 0, p] + Xstatic_lead[:, l] for p in range(Xemb_.shape[2])}
                Xout, varsel_wgts = self.varsel(Xin)  # [B, D], [B, I+1]
                Xout = self.layer_norm3(Xout)  # B, D
                wgts.append(varsel_wgts)
                Xemb = Xout.unsqueeze(1)
                # Xemb = torch.concat((Xemb[:, 1:], Xout[:, None]), 1)  # B, T, D
        yemb = torch.stack(yemb, 1)  # B, L, D
        wgts = torch.stack(wgts, 1).squeeze(2)  # B, L, I+1

        Y = self.head(yemb)  # B, L, N*H*W
        Y = Y.view(B, self.L, -1, H, W)  # B, L, N, H, W
        
        return Y, wgts
    
    def train_step(self, X, xmask, Y, optimizer, criterion, clip, lat_wgts):

        optimizer.zero_grad()
        y, _ = self(X, xmask)  # B, L, N, H, W
        loss = criterion(y.permute(0, 2, 1, 3, 4), Y, lat_wgts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)  # Clip gradients
        optimizer.step()
        return loss.item()
    
    def val_step(self, val_dataloader, criterion, lat_wgts):
        self.eval()
        loss = 0
        for X, xmask, Y in val_dataloader:
            y, _ = self(X, xmask)
            loss += criterion(y[:, 3:].permute(0, 2, 1, 3, 4), Y[:, 3:], lat_wgts).item()
        return loss / len(val_dataloader)

    def train_model(self, train_dataloader, epochs, clip, lr=1e-3, val_dataloader=None, lat_wgts=None):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        criterion = CRPSLoss(-999., True)
        scheduler = ExponentialLR(optimizer, 0.995)
        val_loss = torch.tensor(float('inf'))
        earlystop = EarlyStopping(3, 1e-5)
        best_state = None
        for epoch in range(epochs):
            self.train()
            loss = 0
            for X, xmask, Y in train_dataloader:
                loss += self.train_step(X, xmask, Y, optimizer, criterion, clip, lat_wgts)
            loss /= len(train_dataloader)
            scheduler.step()
            if val_dataloader is not None:
                val_loss = self.val_step(val_dataloader, criterion, lat_wgts[:, 3:])
                earlystop(val_loss)
                if earlystop.counter == 0:
                    best_state = deepcopy(self.state_dict())
                if earlystop.early_stop:
                    self.load_state_dict(best_state)
                    printProgressBar(epoch, epochs, f' - early stopping at epoch {epoch}, train loss = {loss:.4f}, val loss = {val_loss:.4f}', printEnd=None, prefix='Progress:', suffix='Complete', length=50)
                    break
                if epoch == epochs-1:
                    self.load_state_dict(best_state)
            printProgressBar(epoch+1, epochs, f' - epoch = {epoch+1}, train loss = {loss:.4f}, val loss = {val_loss:.4f}', prefix='Progress:', suffix='Complete', length=50)
        return loss

    def inference(self, X, xmask):
        self.eval()
        with torch.no_grad():
            y, wgts = self(X, xmask)
        return y, wgts

