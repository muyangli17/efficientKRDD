import torch


def calculateW1(p, q, pf):
    # 将 p 和 q 拼接为一个大 batch，单次前向传播，减少 GPU kernel launch
    n_p = p.size(0)
    pq = torch.cat([p, q], dim=0)
    fpq = pf(pq)
    fp, fq = fpq[:n_p], fpq[n_p:]  # p、q 的 batch size 可能不同，不能用 chunk
    return torch.mean(fp) - torch.mean(fq)


def potentialFunTrainer(p, q, pf, optimizer, epoch=100, lip_strategy: str = None):
    record = {"W-distance": [], "gradient_penalty": []}

    for ep in range(epoch):
        if lip_strategy == "truncate":
            # 截断参数
            pf.truncate_parameters()
            loss = -calculateW1(p, q, pf)
        if lip_strategy == "SpectralNorm":
            pass

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        record["W-distance"].append(-loss.item())
    return pf, record
