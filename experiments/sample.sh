DOMAIN=wmt21.en-de
MODEL=None # Use "None" for using the sequence-to-sequence models.
PROMPT=None
NLINES=3
STARTITER=0
NSAMPLES=37

EPS=0.02
TOPK=0
TOPP=1.0 # nucleus

BSZ=16

while getopts d:m:p:l:s:e:k:n:t:z:a: option
do
  case $option in
    d)
        DOMAIN=${OPTARG};;
    m)
        MODEL=${OPTARG};;
    p)
        PROMPT=${OPTARG};;
    l)
        NLINES=${OPTARG};;
    s)
        NSAMPLES=${OPTARG};;
    e)
        EPS=${OPTARG};;
    k)
        TOPK=${OPTARG};;
    n)
        TOPP=${OPTARG};;
    z)
        BSZ=${OPTARG};;
    a)
        STARTITER=${OPTARG};;
    \?)
      echo "This is unexpected option." 1>&2
      exit 1
  esac
done

MINBSZ=$(( $BSZ < $NSAMPLES ? $BSZ : $NSAMPLES ))
BSZ=$MINBSZ

echo "sampling..."
python3 mbr/sample.py $DOMAIN \
    --model $MODEL --prompt $PROMPT \
    --n_lines $NLINES --start_iter $STARTITER \
    --n_samples $NSAMPLES \
    --eps $EPS --topk $TOPK --topp $TOPP \
    --bsz $BSZ

