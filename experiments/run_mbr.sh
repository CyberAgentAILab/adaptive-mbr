# Default parameters are set to run a debug experiment.

DOMAIN=wmt21.en-de
MODEL=m2m100_418M
NLINES=3
NSAMPLES=16
EPS=0.02
TOPK=0
TOPP=1.0
SIM=bertscore
EVAL=sacrebleu
ALGORITHM=None
DEBUG=0
RECOMPUTE=""

RZERO=4
PALPHA=0.9

BUDGETS=-1

while getopts d:m:p:l:s:e:k:n:i:v:a:bru:t:z:w:o:h: option
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
        # Nucleus sampling
        TOPP=${OPTARG};;
    i)
        SIM=${OPTARG};;
    v)
        EVAL=${OPTARG};;
    a)
        ALGORITHM=${OPTARG};;
    b)
        DEBUG=1;;
    r)
        RECOMPUTE="--recompute";;
    u)
        # APPROXIMATION
        BUDGETS=${OPTARG};;
    o)
        # APPROXIMATION
        RZERO=${OPTARG};;
    h)
        # APPROXIMATION
        PALPHA=${OPTARG};;
    \?)
      echo "This is unexpected option." 1>&2
      exit 1
  esac
done

if [ "$DOMAIN" == "xsum" ]; then
    DATADIR=xsum
elif [ "$DOMAIN" == "samsum" ]; then
    DATADIR=None
elif [[ $DOMAIN == "wmt21"* ]]; then
    DATADIR=wmt21
elif [ "$DOMAIN" == "mscoco-ft" ]; then
    DATADIR=None
else
    echo "No cache available for $DOMAIN. Loading from huggingface datasets."
    DATADIR=None
fi

set -e

python3 mbr/mbr_engine.py $DOMAIN \
    --model $MODEL \
    --sample_dir ./samples/$DOMAIN/$MODEL \
    --n_lines $NLINES --n_samples $NSAMPLES \
    --eps $EPS --topk $TOPK --topp $TOPP \
    --sim $SIM \
    --eval $EVAL \
    --algorithm $ALGORITHM \
    $RECOMPUTE \
    --approx_budgets $BUDGETS \
    --r_0 $RZERO --pruning_alpha $PALPHA

if [ "$DEBUG" == "1" ]; then
    echo "done!"

fi
