file_name = "/scratch/hb-macocu/NMT_eval/en-is/data/CCM.CCA.Para.Tilde.en-is.dedup.norm.tsv"

corpus_src = []
corpus_tgt = []
error_count = 0
with open(file_name, 'r', encoding="utf-8") as f:
    for line in f:
        try:
            src, tgt = line.strip().split('\t')
            # only select lines with more than 25 words
            if len(src.split()) < 25:
                continue
            corpus_src.append(src)
            corpus_tgt.append(tgt)
        except:
            error_count += 1
            continue

print("Number of lines with errors:", error_count)
print("Number of lines in the corpus:", len(corpus_src))

# save the corpus to a file in the same format as the original corpus
with open("/scratch/s3412768/genre_NMT/en-is/data/CCM.CCA.Para.Tilde.en-is.dedup.norm.25.tsv", "w") as f:
    for i in range(len(corpus_src)):
        f.write("{}\t{}\n".format(corpus_src[i], corpus_tgt[i]))
