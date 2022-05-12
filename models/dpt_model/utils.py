def match_weights(weights):
    for w in weights.copy():
        if w.startswith("pretrained.model"):
            weights[w.replace("pretrained.model", "backbone")] = weights.pop(w)
        if w.startswith("pretrained.act_postprocess"):
            weights[w.replace("pretrained", "backbone")] = weights.pop(w)
        if w.startswith("scratch.output_conv"):
            weights[w.replace("scratch.output_conv", "head")] = weights.pop(w)

    return weights
