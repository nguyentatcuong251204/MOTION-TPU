INPUT_DIR="/data1/hongn/SSv2/20bn-something-something-v2"
OUTPUT_DIR="/data1/hongn/SSv2/frames"
EXTENSION="webm"
count=0
for filepath in "$INPUT_DIR"*/*."$EXTENSION"; do
    if [ "$count" -le 25751 ]; then
        ((count++))
        continue
    fi
    filename="${filepath:${#INPUT_DIR}:${#filepath}-${#INPUT_DIR}-${#EXTENSION}-1}";
    destination="$OUTPUT_DIR$filename";
    mkdir -p "$destination";
    ffmpeg -i "$filepath" -r 30 -q:v 1 "$destination/${filename}_%6d.jpg";
done