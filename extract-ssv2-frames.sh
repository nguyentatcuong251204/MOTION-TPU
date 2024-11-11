INPUT_DIR="/mnt/d/ssv2/videos"
OUTPUT_DIR="/mnt/d/ssv2/frames"
EXTENSION="webm"
for filepath in "$INPUT_DIR"*/*."$EXTENSION"; do
    filename="${filepath:${#INPUT_DIR}:${#filepath}-${#INPUT_DIR}-${#EXTENSION}-1}";
    destination="$OUTPUT_DIR$filename";
    mkdir -p "$destination";
    ffmpeg -i "$filepath" -r 30 -q:v 1 "$destination/${filename}_%6d.jpg";
done