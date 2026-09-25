# Cell 17. Statistical controls for the SWD analysis
# Requires: model, train_loader, test_loader, df_train, df_test, emb_train, emb_test,
#           id2label, DEVICE, swd, LAYERS (cells 1-16).
from transformers import XLMRobertaModel

N_PERM = 500   # permutations (p-value resolution about 0.002)
N_REP = 200    # subsamples for the size-matched null
rng = np.random.default_rng(0)


def entity_embs(loader, encoder, ent_type, layer):
    """One vector per gold entity of ent_type: mean over all its subword tokens."""
    vecs = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            out = encoder(input_ids=ids, attention_mask=attn, output_hidden_states=True)
            H = out.hidden_states[layer].cpu().numpy()
            labels = batch["labels"]
            for i in range(labels.size(0)):
                pos = [j for j in range(labels.size(1)) if labels[i][j] != -100]
                tags = [id2label[labels[i][j].item()] for j in pos]
                seq_end = int(attn[i].sum().item()) - 1          # exclude </s>
                for t, s, e in get_entities(tags):
                    if t != ent_type:
                        continue
                    end = pos[e + 1] if e + 1 < len(pos) else seq_end
                    vecs.append(H[i, pos[s]:end].mean(axis=0))
    return np.array(vecs)


def perm_test(A, B, n_perm=N_PERM):
    """Observed SWD, mean SWD under random re-splits of the pooled data, and p-value."""
    obs = swd(A, B)
    pooled = np.vstack([A, B])
    na = len(A)
    null = np.empty(n_perm)
    for k in range(n_perm):
        idx = rng.permutation(len(pooled))
        null[k] = swd(pooled[idx[:na]], pooled[idx[na:]])
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, null.mean(), p


def size_matched(A_big, B_big, na, nb, n_rep=N_REP):
    """SWD of a well-represented class on subsamples of the given sizes."""
    return np.array([
        swd(A_big[rng.choice(len(A_big), na, replace=False)],
            B_big[rng.choice(len(B_big), nb, replace=False)])
        for _ in range(n_rep)
    ])


frozen = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(DEVICE)

rows = []
for enc_name, enc in [("fine-tuned", model), ("frozen", frozen)]:
    for layer in LAYERS:
        E = {}
        for split, ld in [("train", train_loader), ("test", test_loader)]:
            for t in ["STOP_NAME", "STREET_NAME"]:
                E[(split, t)] = entity_embs(ld, enc, t, layer)
        st_tr, st_te = E[("train", "STREET_NAME")], E[("test", "STREET_NAME")]
        sp_tr, sp_te = E[("train", "STOP_NAME")], E[("test", "STOP_NAME")]

        obs_st, null_st, p_st = perm_test(st_tr, st_te)
        obs_sp, null_sp, p_sp = perm_test(sp_tr, sp_te)
        matched = size_matched(sp_tr, sp_te, len(st_tr), len(st_te))

        rows.append({
            "encoder": enc_name, "layer": layer,
            "n_street(tr/te)": f"{len(st_tr)}/{len(st_te)}",
            "SWD_street": round(obs_st, 4), "perm_null_street": round(null_st, 4), "p_street": round(p_st, 4),
            "SWD_stop": round(obs_sp, 4), "perm_null_stop": round(null_sp, 4), "p_stop": round(p_sp, 4),
            "stop_size_matched_mean": round(matched.mean(), 4),
            "stop_size_matched_95%": round(np.percentile(matched, 95), 4),
            "street_above_matched_%": round(100 * (matched < obs_st).mean(), 1),
        })

res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res.to_string(index=False))
res.to_csv("swd_checks.csv", index=False)

# Language composition of complaints with STREET_NAME
print("\nLanguage of complaints with STREET_NAME (shares):")
for split, d in [("train", df_train), ("test", df_test)]:
    sub = d[d.has_STREET_NAME == 1]
    print(f"{split:5s} n={len(sub):4d}", sub["lang"].value_counts(normalize=True).round(2).to_dict())

# STREET_NAME shift within Russian-only complaints: if it stays large,
# language mixing is not the main cause.
m_tr = ((df_train.has_STREET_NAME == 1) & (df_train.lang == "ru")).values
m_te = ((df_test.has_STREET_NAME == 1) & (df_test.lang == "ru")).values
if m_tr.sum() >= 5 and m_te.sum() >= 5:
    obs, null_mean, p = perm_test(emb_train[12][m_tr], emb_test[12][m_te])
    print(f"\nru-only STREET complaints, layer 12: SWD={obs:.4f}, permutation null={null_mean:.4f}, p={p:.4f}")
else:
    print("\nru-only STREET complaints: too few examples for a test")

# How to read the results:
#   p_street < 0.05              -> the train-test shift for street names is significant
#   street_above_matched_% > 95  -> it exceeds STOP_NAME at the same sample size,
#                                   so it is not only a small-sample effect
#   shift also with frozen       -> the shift is in the data, not only memorization
#   fine-tuned >> frozen         -> most of the "shift" comes from fitting the train set
#   ru-only also significant     -> language mixing is not the main cause
