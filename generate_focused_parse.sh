LANG="
basque
chinese
english
french
german
hebrew
hungarian
korean
polish
swedish
"

TYPE="
left-branched
right-branched
random
left-binarized
right-binarized
"

DATASET="
train
valid
test
"

for lang in $LANG; do
    for type in $TYPE; do
        for dataset in $DATASET; do
            python -m preprocessing.generate_focused_parse \
            --factor ${type} \
            --vocab "vocab/${lang}.vocab" \
            --input "data/data.clean/${lang}-${dataset}.txt" \
            --output "trees/${type}_${lang}_${dataset}.pt"
        done
    done
done