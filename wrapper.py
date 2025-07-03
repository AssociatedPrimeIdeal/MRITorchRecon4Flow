from utils import *

def ReconLLR(kspc, csm, reg, blk=-1, itmethod='POGM', it=10, Lc=2., device='cuda', verbose=0, stopth=1e-3):
    """
    Perform LLR based reconstruction.

    Args:
        kspc : The k-space data with shape [Nv, Nt, Nc, FE, PE, SPE] (Velocity Encoding, Temporal, Coils, Frequency Encoding, Phase Encoding, Slice Phase Encoding).
        csm : Coil sensitivity maps with shape [Nc, FE, PE, SPE].
        reg : Regularization parameters for Low-rank term.
        itmethod : Iterative method to use for reconstruction ('ISTA', 'FISTA', 'POGM')
        it : Number of iterations for the reconstruction algorithm.
        Lc : Lipschitz constant. 
        device : The device to perform the computations on (e.g., 'cpu', 'cuda').
        verbose : Verbosity level, where 0 means silent. Defaults to 0.
        stopth : Stopping threshold for iterative convergence. Defines the minimum relative change in reconstruction error required to continue iterations.
    """
    Nv, Nt, Nc, FE, PE, SPE = kspc.shape
    kspc = torch.as_tensor(np.ascontiguousarray(kspc)).to(torch.complex64).to(device)
    csm = torch.as_tensor(np.ascontiguousarray(csm)).to(torch.complex64).to(device)
    us_mask = (torch.abs(kspc[:, :, 0:1, :]) > 0).to(torch.float32).to(device)
    print("LOAD KSPC SHAPE", kspc.shape, "LOAD CSM SHAPE", csm.shape, 'US RATE:', 1/torch.mean(us_mask))
    rcomb = torch.sum(k2i_torch(kspc, ax=[-3, -2, -1]) * torch.conj(csm), -4)
    regFactor = torch.max(torch.abs(rcomb))
    kspc /= regFactor
    loss_prev = None
    A = Eop(csm, us_mask)
    loop = tqdm.tqdm(range(1, it + 1), total=it)
    if blk == -1:
        llr = 0
    else:
        llr = 1
    if llr:
        stepx = np.ceil(FE / blk)
        stepy = np.ceil(PE / blk)
        stepz = np.ceil(SPE / blk)
        padx = (stepx * blk).astype('int64')
        pady = (stepy * blk).astype('int64')
        padz = (stepz * blk).astype('int64')
        M = blk ** 3
        N = Nt
        B = padx * pady * padz / M
        RF = GETWIDTH(M, N, B)
        reg *= RF
    else:
        reg *= (np.sqrt(np.prod(kspc.shape[-3:])) + 1)
    if itmethod == 'ISTA':
        X = A.mtimes(kspc, 1)
        for i in loop:
            if verbose:
                print(" ")
            axb = A.mtimes(X, 0) - kspc
            X = X - 1 / Lc * A.mtimes(axb, 1)
            if llr:
                X, loss_LR = SVT_LLR(X, reg / Lc, blk)
            else:
                X, loss_LR = SVT(X, reg / Lc)
            loss_LR *= reg
            loss_DC = torch.sum(torch.abs(A.mtimes(X, 0) - kspc) ** 2).item() * 0.5
            loss = loss_DC + loss_LR
            loop.set_postfix({'loss': '{:.5f}'.format(loss),'loss_DC': '{:.5f}'.format(loss_DC), 'loss_LR': '{:.5f}'.format(loss_LR)})  
            if loss_prev is not None:
                relerr = np.abs(loss - loss_prev) / np.abs(loss_prev)
                if np.abs(relerr) < stopth:
                    break
            loss_prev = loss
    elif itmethod == 'FISTA':
        tp = 1
        Xp = A.mtimes(kspc, 1)
        Y = Xp.clone()
        for i in loop:
            if verbose:
                print(" ")
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            axb = A.mtimes(Y, 0) - kspc
            Y = Y - 1 / Lc * A.mtimes(axb, 1)
            if llr:
                X, loss_LR = SVT_LLR(Y, reg / Lc, blk)
            else:
                X, loss_LR = SVT(Y, reg / Lc)
            Y = X + (tp - 1) / t * (X - Xp)
            Xp = X
            tp = t
            loss_LR *= reg
            loss_DC = torch.sum(torch.abs(A.mtimes(X, 0) - kspc) ** 2).item() * 0.5
            loss = loss_DC + loss_LR
            loop.set_postfix({'loss': '{:.5f}'.format(loss),'loss_DC': '{:.5f}'.format(loss_DC), 'loss_LR': '{:.5f}'.format(loss_LR)})  
            if loss_prev is not None:
                relerr = np.abs(loss - loss_prev) / np.abs(loss_prev)
                if np.abs(relerr) < stopth:
                    break
            loss_prev = loss
    elif itmethod == 'POGM':
        tp = 1
        gp = 1
        Xp = A.mtimes(kspc, 1)
        X, Y, Z, Yp, Zp = Xp.clone(), Xp.clone(), Xp.clone(), Xp.clone(), Xp.clone()
        for i in loop:
            if verbose:
                print(" ")
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            g = 1 / Lc * (2 * tp + t - 1) / t
            axb = A.mtimes(X, 0) - kspc
            Y = X - 1 / Lc * A.mtimes(axb, 1)
            Z = Y + (tp - 1) / t * (Y - Yp) + tp / t * (Y - Xp) + (tp - 1) / (Lc * gp * t) * (Zp - Xp)
            if llr:
                X, loss_LR = SVT_LLR(Z, reg * g, blk)
            else:
                X, loss_LR = SVT(Z, reg * g)
            Xp = X
            Yp = Y
            Zp = Z
            tp = t
            gp = g
            loss_LR *= reg
            loss_DC = torch.sum(torch.abs(A.mtimes(X, 0) - kspc) ** 2).item() * 0.5
            loss = loss_DC + loss_LR
            loop.set_postfix({'loss': '{:.5f}'.format(loss),'loss_DC': '{:.5f}'.format(loss_DC), 'loss_LR': '{:.5f}'.format(loss_LR)})  
            if loss_prev is not None:
                relerr = np.abs(loss - loss_prev) / np.abs(loss_prev)
                if np.abs(relerr) < stopth:
                    break
            loss_prev = loss
    return X * regFactor


def ReconLplusS(kspc, csm, regL, regS, blk=-1,HADAMARD=0, itmethod='POGM', it=10, Lc=2., device='cuda', verbose=0, stopth=1e-3):
    """
    Perform L+S based reconstruction.

    Args:
        kspc : The k-space data with shape [Nv, Nt, Nc, FE, PE, SPE] (Velocity Encoding, Temporal, Coils, Frequency Encoding, Phase Encoding, Slice Phase Encoding).
        csm : Coil sensitivity maps with shape [Nc, FE, PE, SPE].
        regL : Regularization parameters for Low-rank term.
        regS : Regularization parameters for Sparse term.
        HADAMARD : Perform HADAMARD transform before sparse transform. (https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.26508)
        itmethod : Iterative method to use for reconstruction ('ISTA', 'FISTA', 'POGM')
        it : Number of iterations for the reconstruction algorithm.
        Lc : Lipschitz constant. 
        device : The device to perform the computations on (e.g., 'cpu', 'cuda').
        verbose : Verbosity level, where 0 means silent. Defaults to 0.
        stopth : Stopping threshold for iterative convergence. Defines the minimum relative change in reconstruction error required to continue iterations.
    """
    Nv, Nt, Nc, FE, PE, SPE = kspc.shape
    kspc = torch.as_tensor(np.ascontiguousarray(kspc)).to(torch.complex64).to(device)
    csm = torch.as_tensor(np.ascontiguousarray(csm)).to(torch.complex64).to(device)
    us_mask = (torch.abs(kspc[:, :, 0:1, FE // 2:FE // 2 + 1]) > 0).to(torch.float32).to(device)
    print("LOAD KSPC SHAPE", kspc.shape, "LOAD CSM SHAPE", csm.shape, 'US RATE:', 1/torch.mean(us_mask))
    rcomb = torch.sum(k2i_torch(kspc, ax=[-3, -2, -1]) * torch.conj(csm), -4)
    regFactor = torch.max(torch.abs(rcomb))
    kspc /= regFactor
    loss_prev = None
    if Nv == 4 and HADAMARD:
        HADAMARD = 1
    else:
        HADAMARD = 0
    A = Eop(csm, us_mask)
    loop = tqdm.tqdm(range(1, it + 1), total=it)
    if blk == -1:
        llr = 0
    else:
        llr = 1
    if llr:
        stepx = np.ceil(FE / blk)
        stepy = np.ceil(PE / blk)
        stepz = np.ceil(SPE / blk)
        padx = (stepx * blk).astype('int64')
        pady = (stepy * blk).astype('int64')
        padz = (stepz * blk).astype('int64')
        M = blk ** 3
        N = Nt
        B = padx * pady * padz / M
        RF = GETWIDTH(M, N, B)
        regL *= RF
    else:
        regL *= (np.sqrt(np.prod(kspc.shape[-3:])) + 1)

    if itmethod == 'ISTA':
        X = A.mtimes(kspc, 1)
        L, Lp = X.clone(), X.clone()
        S = torch.zeros_like(X).to(device)
        for i in loop:
            if verbose:
                print(" ")
            if llr:
                L, loss_LR = SVT_LLR(X - S, regL / Lc, blk)
            else:
                L, loss_LR = SVT(X - S, regL / Lc)
            S, loss_S = Sparse(X - Lp, regS / Lc, 1, HADAMARD)
            axb = A.mtimes(L + S, 0) - kspc
            X = L + S - 1 / Lc * A.mtimes(axb, 1)
            Lp = L
            loss_DC = torch.sum(torch.abs(A.mtimes(X, 0) - kspc) ** 2).item() * 0.5
            loss_S *= regS
            loss_LR *= regL
            loss = loss_DC + loss_LR + loss_S
            loop.set_postfix({'loss': '{:.5f}'.format(loss),'loss_DC': '{:.5f}'.format(loss_DC), 'loss_LR': '{:.5f}'.format(loss_LR), 'loss_S': '{:.5f}'.format(loss_S)}) 
            if loss_prev is not None:
                relerr = np.abs(loss - loss_prev) / np.abs(loss_prev)
                if np.abs(relerr) < stopth:
                    break
            loss_prev = loss 
    elif itmethod == 'FISTA':
        tp = 1
        X = A.mtimes(kspc, 1)
        L, Lp, Lh = X.clone(), X.clone(), X.clone()
        S, Sp, Sh = torch.zeros_like(X).to(device), torch.zeros_like(X).to(device), torch.zeros_like(X).to(device)
        for i in loop:
            if verbose:
                print(" ")
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            if llr:
                L, loss_LR = SVT_LLR(X - Sh, regL / Lc, blk)
            else:
                L, loss_LR = SVT(X - Sh, regL / Lc)
            S, loss_S = Sparse(X - Lh, regS / Lc, 1, HADAMARD)
            Lh = L + (tp - 1) / t * (L - Lp)
            Sh = S + (tp - 1) / t * (S - Sp)
            axb = A.mtimes(Lh + Sh, 0) - kspc
            X = Lh + Sh - 1 / Lc * A.mtimes(axb, 1)
            tp = t
            Lp = L
            Sp = S
            loss_DC = torch.sum(torch.abs(A.mtimes(X, 0) - kspc) ** 2).item() * 0.5
            loss_S *= regS
            loss_LR *= regL
            loss = loss_DC + loss_LR + loss_S
            loop.set_postfix({'loss': '{:.5f}'.format(loss),'loss_DC': '{:.5f}'.format(loss_DC), 'loss_LR': '{:.5f}'.format(loss_LR), 'loss_S': '{:.5f}'.format(loss_S)}) 
            if loss_prev is not None:
                relerr = np.abs(loss - loss_prev) / np.abs(loss_prev)
                if np.abs(relerr) < stopth:
                    break
            loss_prev = loss   
    elif itmethod == 'POGM':
        tp = 1
        gp = 1
        X = A.mtimes(kspc, 1)
        L, L_, L_p, Lh, Lhp = X.clone(), X.clone(), X.clone(), X.clone(), X.clone()
        S, S_, S_p, Sh, Shp = torch.zeros_like(X).to(device), torch.zeros_like(X).to(device), torch.zeros_like(X).to(
            device), torch.zeros_like(X).to(device), torch.zeros_like(X).to(device)
        for i in loop:
            if verbose:
                print(" ")
            Lh = X - S
            Sh = X - L
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            L_ = Lh + (tp - 1) / t * (Lh - Lhp) + tp / t * (Lh - L) + (tp - 1) / (gp * t) * 1 / Lc * (L_p - L)
            S_ = Sh + (tp - 1) / t * (Sh - Shp) + tp / t * (Sh - S) + (tp - 1) / (gp * t) * 1 / Lc * (S_p - S)
            g = 1 / Lc * (1 + (tp - 1) / t + tp / t)
            if llr:
                L, loss_LR = SVT_LLR(L_, regL * g, blk)
            else:
                L, loss_LR = SVT(L_, regL * g)
            S, loss_S = Sparse(S_, regS * g, 1, HADAMARD)
            axb = A.mtimes(L + S, 0) - kspc
            X = L + S - 1 / Lc * A.mtimes(axb, 1)
            tp = t
            gp = g
            L_p = L_
            S_p = S_
            Lhp = Lh
            Shp = Sh
            loss_DC = torch.sum(torch.abs(A.mtimes(X, 0) - kspc) ** 2).item() * 0.5
            loss_S *= regS
            loss_LR *= regL
            loss = loss_DC + loss_LR + loss_S
            loop.set_postfix({'loss': '{:.5f}'.format(loss),'loss_DC': '{:.5f}'.format(loss_DC), 'loss_LR': '{:.5f}'.format(loss_LR), 'loss_S': '{:.5f}'.format(loss_S)}) 
            if loss_prev is not None:
                relerr = np.abs(loss - loss_prev) / np.abs(loss_prev)
                if np.abs(relerr) < stopth:
                    break
            loss_prev = loss     
    return X * regFactor

def ReconHAAR(kspc, csm, reg_list, itmethod='POGM', it=10, Lc=1., device='cuda', verbose=0, stopth=1e-3):
    """
    Perform HAAR Wavelet based reconstruction.

    Args:
        kspc : The k-space data with shape [Nv, Nt, Nc, FE, PE, SPE] (Velocity Encoding, Temporal, Coils, Frequency Encoding, Phase Encoding, Slice Phase Encoding).
        csm : Coil sensitivity maps with shape [Nc, FE, PE, SPE].
        reg_list : List of regularization parameters for haar transform.
        itmethod : Iterative method to use for reconstruction ('ISTA', 'FISTA', 'POGM')
        it : Number of iterations for the reconstruction algorithm.
        Lc : Lipschitz constant. 
        device : The device to perform the computations on (e.g., 'cpu', 'cuda').
        verbose : Verbosity level, where 0 means silent. Defaults to 0.
        stopth : Stopping threshold for iterative convergence. Defines the minimum relative change in reconstruction error required to continue iterations.
    """
    Nv, Nt, Nc, FE, PE, SPE = kspc.shape
    kspc = torch.as_tensor(np.ascontiguousarray(kspc)).to(torch.complex64).to(device)
    csm = torch.as_tensor(np.ascontiguousarray(csm)).to(torch.complex64).to(device)
    us_mask = (torch.abs(kspc[:, :, 0:1, FE // 2:FE // 2 + 1]) > 0).to(torch.float32).to(device)
    print("LOAD KSPC SHAPE", kspc.shape, "LOAD CSM SHAPE", csm.shape, 'US RATE:', 1/torch.mean(us_mask))
    rcomb = torch.sum(k2i_torch(kspc, ax=[-3, -2, -1]) * torch.conj(csm), -4)
    regFactor = torch.max(torch.abs(rcomb))
    kspc /= regFactor
    loss_prev = None
    A = Eop(csm, us_mask)
    loop = tqdm.tqdm(range(1, it + 1), total=it)
    if itmethod == 'ISTA':
        X = A.mtimes(kspc, 1)
        for i in loop:
            if verbose:
                print(" ")
            axb = A.mtimes(X, 0) - kspc
            X = X - 1 / Lc * A.mtimes(axb, 1)
            X, loss_H = ST_HAAR(X, reg_list / Lc, device)
            loss_DC = torch.sum(torch.abs(A.mtimes(X, 0) - kspc) ** 2).item() * 0.5
            loss = loss_DC + loss_H
            loop.set_postfix({'loss': '{:.5f}'.format(loss), 'loss_DC': '{:.5f}'.format(loss_DC), 'loss_H': '{:.5f}'.format(loss_H)})  
            if loss_prev is not None:
                relerr = np.abs(loss - loss_prev) / np.abs(loss_prev)
                if np.abs(relerr) < stopth:
                    break
            loss_prev = loss 

    elif itmethod == 'FISTA':
        tp = 1
        Xp = A.mtimes(kspc, 1)
        Y = Xp.clone()
        for i in loop:
            if verbose:
                print(" ")
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            axb = A.mtimes(Y, 0) - kspc
            Y = Y - 1 / Lc * A.mtimes(axb, 1)
            X, loss_H = ST_HAAR(Y, reg_list / Lc, device)
            Y = X + (tp - 1) / t * (X - Xp)
            Xp = X
            tp = t
            loss_DC = torch.sum(torch.abs(A.mtimes(X, 0) - kspc) ** 2).item() * 0.5
            loss = loss_DC + loss_H
            loop.set_postfix({'loss': '{:.5f}'.format(loss), 'loss_DC': '{:.5f}'.format(loss_DC), 'loss_H': '{:.5f}'.format(loss_H)})  
            if loss_prev is not None:
                relerr = np.abs(loss - loss_prev) / np.abs(loss_prev)
                if np.abs(relerr) < stopth:
                    break
            loss_prev = loss  
    elif itmethod == 'POGM':
        tp = 1
        gp = 1
        Xp = A.mtimes(kspc, 1)
        X, Y, Z, Yp, Zp = Xp.clone(), Xp.clone(), Xp.clone(), Xp.clone(), Xp.clone()
        for i in loop:
            if verbose:
                print(" ")
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            g = 1 / Lc * (2 * tp + t - 1) / t
            axb = A.mtimes(X, 0) - kspc
            Y = X - 1 / Lc * A.mtimes(axb, 1)
            Z = Y + (tp - 1) / t * (Y - Yp) + tp / t * (Y - Xp) + (tp - 1) / (Lc * gp * t) * (Zp - Xp)
            X, loss_H = ST_HAAR(Z,  reg_list * g , device)
            Xp = X
            Yp = Y
            Zp = Z
            tp = t
            gp = g
            loss_DC = torch.sum(torch.abs(A.mtimes(X, 0) - kspc) ** 2).item() * 0.5
            loss = loss_DC + loss_H
            loop.set_postfix({'loss': '{:.5f}'.format(loss), 'loss_DC': '{:.5f}'.format(loss_DC), 'loss_H': '{:.5f}'.format(loss_H)})  
            if loss_prev is not None:
                relerr = np.abs(loss - loss_prev) / np.abs(loss_prev)
                if np.abs(relerr) < stopth:
                    break
            loss_prev = loss  
    return X


def ReconHAAR_CORE(params):
    torch.backends.cudnn.benchmark = True 
    torch.autograd.profiler.emit_nvtx(enabled=True) 
    torch.set_num_threads(os.cpu_count())
    def shrink1(s, alph, p, ep=1e-10):
        t = torch.abs(s)
        if isinstance(alph, torch.Tensor):
            if alph.dim() < s.dim():
                num_new_dims = s.dim() - alph.dim()
                view_shape = [-1] + [1] * num_new_dims
                alph = alph.view(*view_shape)
        scale = torch.clamp_min(t - alph * (t ** 2 + ep) ** (p / 2 - 0.5), 0)
        w = scale * (s / (t + ep)) 
        return w

    def HAAR4D(X, scale=1 / 2, forward=True, device='cuda'):
        if forward:
            FE, PE, SPE, Nt = X.shape
            X = X.view(FE, PE, SPE, Nt)
            sign = [-1, 1]

            FE, PE, SPE, Nt = X.shape
            sign_comb = torch.tensor(list(itertools.product([-1, 1], repeat=4)), device=device)
            d1, d2, d3, d4 = sign_comb.unbind(dim=1)
            X = X.unsqueeze(0)  
            d1 = d1.view(-1, 1, 1, 1, 1) 
            d2 = d2.view(-1, 1, 1, 1, 1)
            d3 = d3.view(-1, 1, 1, 1, 1)
            d4 = d4.view(-1, 1, 1, 1, 1)

            tmp = scale * (X + d1 * torch.roll(X, shifts=-1, dims=1))
            tmp1 = scale * (tmp + d2 * torch.roll(tmp, shifts=-1, dims=2))
            tmp = scale * (tmp1 + d3 * torch.roll(tmp1, shifts=-1, dims=3))
            y = scale * (tmp + d4 * torch.roll(tmp, shifts=-1, dims=4))

            return y
        else:
            Ch, FE, PE, SPE, Nt = X.shape

            sign_comb = torch.tensor(list(itertools.product([-1, 1], repeat=4)), device=device)
            d1, d2, d3, d4 = sign_comb.unbind(dim=1) 

            d1 = d1.view(-1, 1, 1, 1, 1)  
            d2 = d2.view(-1, 1, 1, 1, 1)
            d3 = d3.view(-1, 1, 1, 1, 1)
            d4 = d4.view(-1, 1, 1, 1, 1)

            tmp = scale * (X + d1 * torch.roll(X, shifts=1, dims=1))
            tmp1 = scale * (tmp + d2 * torch.roll(tmp, shifts=1, dims=2))
            tmp = scale * (tmp1 + d3 * torch.roll(tmp1, shifts=1, dims=3))
            y = scale * (tmp + d4 * torch.roll(tmp, shifts=1, dims=4))
            y = torch.sum(y, 0) 
            return y

    y = params['K']
    A = params['op']
    oIter = params['oit']
    iIter = params['iit']
    gStp = params['gstp']
    mu1 = params['mu1']
    mu2 = params['mu2']
    lam2 = params['lam2']
    dev = params['device']
    lam1 = torch.from_numpy(params['lam1']).to(device=dev, dtype=torch.float32)
    FE, PE, SPE, Nc, Nt = y.shape

    print(dev, y.shape)
    if 'u' in params.keys():
        print("USE U ")
        u = params['u']
    else:
        u = A.mtimes(y, 1)

    v = torch.zeros_like(y).contiguous()
    d1 = torch.zeros((16, FE, PE, SPE, Nt)).to(torch.complex64).to(dev)
    b1 = d1.clone()
    d2 = v.reshape(-1).clone()
    b2 = v.reshape(-1).clone()
    v = v.contiguous()
    for i in range(oIter):
        for j in range(iIter):
            gradA = 2 * A.mtimes((A.mtimes(u, 0) + v.reshape(FE, PE, SPE, Nc, Nt) - y), 1)
            gradW = mu1 * HAAR4D((HAAR4D(u, forward=True, device=dev) - d1.reshape(16, FE,
                                                                                                               PE, SPE,
                                                                                                               Nt) + b1.reshape(
                16,

                FE,
                PE,
                SPE,
                Nt)),
                                 forward=False, device=dev)
            u -= gStp * (gradA + gradW)

        Au = A.mtimes(u, 0)
        for j in range(iIter):
            gradA = Au + v.reshape(FE, PE, SPE, Nc, Nt) - y
            v = v.reshape(FE, -1)
            gradW = mu2 * (v.reshape(-1) - (d2 - b2) * (
                    v / (torch.sqrt(torch.sum(torch.abs(v) ** 2, axis=0)) + 1e-6)).reshape(-1))
            v = v.reshape(-1) - gStp * (gradA.reshape(-1) + gradW)
        del gradA
        del gradW
        # Update auxiliary variables
        Wdecu = HAAR4D(u, forward=True, device=dev)

        d1 = shrink1(Wdecu + b1, lam1 / mu1, 1)

        b1 += (Wdecu - d1)
        v = v.reshape(FE, -1)
        b2 = b2.reshape(FE, -1)
        v_norm = torch.linalg.norm(v, dim=0, ord=2) 
        d2 = shrink1(v_norm + b2, lam2 / mu2, 1)
        b2 += (v_norm - d2)
        b2 = b2.reshape(-1)
        d2 = d2.reshape(-1)
        objA = 0.5 * torch.sum(torch.abs(Au + v.reshape(FE, PE, SPE, Nc, Nt) - y) ** 2)
        objW = torch.sum(torch.abs(lam1.reshape(16, -1) * Wdecu.reshape(16, -1)))
        objV = torch.sum(lam2 * torch.sqrt(torch.sum(torch.abs(v.reshape(FE, -1)) ** 2, dim=1)))
        obj = objA + objW + objV

        print(
            f'CORe: Iter = {i} \tobjA= {objA.item():.2f}\tobjW= {objW.item():.2f}\tobjv= {objV.item():.2f}\ttotal_obj= {obj.item():.2f}\t')

        del Au
        del Wdecu


    return u.cpu().numpy(), v.view(FE, PE, SPE, Nc, Nt).cpu().numpy()

