mkdir models; 
cd models;
_FILE_NAME="tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

[ -f "$_FILE_NAME" ] || [ -d "$__FILE_NAME" ] && {
 	1>&2 echo file or folder exists
	exit 1;
}

_TO_DOWNLOAD=https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/"$_FILE_NAME"

wget -O "$_FILE_NAME" "$_TO_DOWNLOAD"
