# GGUF quantization (Q8_0, Q4_K_M) with llama.cpp. Run on any Linux box.
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
pip install gguf sentencepiece
cmake -S /tmp/llama.cpp -B /tmp/lbuild -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/lbuild --target llama-quantize -j4
python /tmp/llama.cpp/convert_hf_to_gguf.py merged_16bit --outfile f16.gguf --outtype f16
/tmp/lbuild/bin/llama-quantize f16.gguf q8_0.gguf Q8_0
/tmp/lbuild/bin/llama-quantize f16.gguf q4_k_m.gguf Q4_K_M
