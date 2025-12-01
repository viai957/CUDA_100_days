#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

#define block_size 32

// ==========================================
// Helper Functions
// ==========================================

void read_csv(float *inp, std::string name){
    std::ifstream file(name);
    std::string line;

    while(std::getline(file, line, '\n')){
        *inp = std::stof(line);
        inp++;
    }
}

void init_zero(float *a, int n){
    for (int i=0; i<n; i++){
        a[i] = 0.0f;
    }
}

void set_eq(float *a, float *b, int n){
    for (int i=0; i<n; i++){
        a[i] = b[i];
    }
}

void kaiming_init(float *w, int n_in, int n_out){
    float std = sqrt(2.0f/(float) n_in);
    
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::normal_distribution<float> dist(0.0f, std); 

    for (int i=0; i<n_in*n_out; i++){
        w[i] = dist(gen);
    }
}

// ==========================================
// Base Module Class
// ==========================================

class Module{
    public:
        float *inp, *out;
        int sz_out;
        
        virtual void forward(float *inp, float *out){};
        virtual void backward(){};
        virtual void update(){};
};

// ==========================================
// Kernels
// ==========================================

__gloabl__
void linear_forward(float *input, float *weights, float *bias, float *out, int batch_size, int n_in, int n_out){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    int idx_input, idx_weights, idx_output;

    if ((row < batch_size) && (col < n_out)){
        idx_output = row * n_out = col; // out[row][col]
        out[idx_output] = bias[col]; // out[row][col] = bias[col]

        for (int i = 0; i < n_in; i++){
            idx_input = row * n_in + i; // input[row][i]
            idx_weights = i * n_out + col; // weights[i][col]
            out[idx_output] += input[idx_input] * weights[idx_weights];
        }
    }
}

__gloabal__
void linear_backward(float *input, float *weights, float *out, int batch_size, int n_input, int n_output){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_input, idx_weights, idx_output;

    if ((row < batch_size) && (col < n_output)){
        idx_output = row * n_output + col; // out[row][col]

        for (int i = 0; i < n_input; i++){
            idx_input = row * n_input + i; // input[row][i]
            idx_weights = i * n_output + col; // weights[i][col]
            automicAdd(&input[idx_input], weights[idx_weights] * out[idx_output]);
        }
    }
}


__global__
void linear_update(float *input, float *weights, float *bias, float *out, int batch_size, int n_in, int n_out, float learning_rate){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if ((row < batch_size) && (col < n_out)){
        idx_output = row * n_out + col; // out[row][col]
        atomicAdd(&bias[col], learning_rate * out[idx_output]);

        for (int i = 0; i < n_in; i++){
            idx_input = row * n_in + i; // input[row][i]
            idx_weights = i * n_out + col; // weights[i][col]
            atomicAdd(&weights[idx_weights], -learning_rate * input[idx_input] * out[idx_output]); // weights[i][col] += learning_rate * input[row][i] * out[row][col]
        }
    }
}

__global__
void relu_forward(float *input, float *output, int sz_out){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < sz_out){
        output[idx] = (0 < input[idx]) * input[idx]; // output[idx] = max(0, input[idx])
    }
}

__global__
void relu_backward(float *input, float *output, int sz_out){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < sz_out){
        inp[ind] = (0 < inp[ind]) * out[idx]; // input[idx] = (0 < input[idx]) * output[idx]
    }
}

__global__
void mse_forward(float *input, float *target, float *out, int sz_out){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < sz_out){
        out[idx] = (input[idx] - target[idx]) * (input[idx] - target[idx]); // out[idx] = (input[idx] - target[idx]) ^ 2
    }
}

__global__
void mse_backward(float *input, float *target, float *out, int sz_out){
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < sz_out){
        out[idx] = 2 * (input[idx] - target[idx]); // out[idx] = 2 * (input[idx] - target[idx])
    }
}

// ==========================================
// Layers
// ==========================================

class Linear_GPU : public Module{
    public:
        int bs, n_in, n_out, sz_weights;
        int n_block_rows, n_block_cols;
        float lr;
        float *weights, *bias, *cp_weights;

        Linear_GPU(int _bs, int _n_in, int _n_out, float _lr=0.01){
            bs = _bs;
            n_in = _n_in;
            n_out = _n_out;
            lr = _lr;

            sz_weights = n_in*n_out;
            sz_out = bs*n_out;
            n_block_rows = (bs + block_size - 1) / block_size;
            n_block_cols = (n_out + block_size - 1) / block_size;

            cudaMallocManaged(&weights, sz_weights*sizeof(float));
            cudaMallocManaged(&bias, n_out*sizeof(float));

            kaiming_init(weights, n_in, n_out);
            init_zero(bias, n_out);
        }

        void forward(float *_inp, float *_out){
            inp = _inp;
            out = _out;

            dim3 n_blocks(n_block_rows, n_block_cols);
            dim3 n_threads(block_size, block_size);

            linear_forward_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out);
            cudaDeviceSynchronize();
        }

        void backward(){
            init_zero(inp, bs*n_in);

            dim3 n_blocks(n_block_rows, n_block_cols);
            dim3 n_threads(block_size, block_size);

            linear_backward_gpu<<<n_blocks, n_threads>>>(inp, cp_weights, out, bs, n_in, n_out);
            cudaDeviceSynchronize();

            cudaFree(cp_weights);
            cudaFree(out);
        }

        void update(){
            cudaMallocManaged(&cp_weights, sz_weights*sizeof(float));
            set_eq(cp_weights, weights, sz_weights);

            dim3 n_blocks(n_block_rows, n_block_cols);
            dim3 n_threads(block_size, block_size);

            linear_update_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out, lr);
            cudaDeviceSynchronize();
        }
};

class ReLU_GPU : public Module{
    public:
        int n_blocks;

        ReLU_GPU(int _sz_out){
            sz_out = _sz_out;
            n_blocks = (sz_out + block_size - 1) / block_size;
        }

        void forward(float *_inp, float *_out){
            inp = _inp;
            out = _out;

            relu_forward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
            cudaDeviceSynchronize();
        }

        void backward(){
            relu_backward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
            cudaDeviceSynchronize();

            cudaFree(out);
        }
};

class MSE_GPU : public Module{
    public:
        int n_blocks;

        MSE_GPU(int _sz_out){
            sz_out = _sz_out;
            n_blocks = (sz_out + block_size - 1) / block_size;
        }

        void forward(float *_inp, float *_out){
            inp = _inp;
            out = _out;
        }

        void _forward(float *_inp, float *_out){
            _out[sz_out] = 0.0f;

            mse_forward_gpu<<<n_blocks, block_size>>>(_inp, _out, sz_out);
            cudaDeviceSynchronize();
        }

        void backward(){
            mse_backward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
            cudaDeviceSynchronize();
        }
};

class Sequential_GPU{
    public:
        std::vector<Module*> layers;

        Sequential_GPU(std::vector<Module*> _layers){
            layers = _layers;
        }

        void forward(float *inp, float *out){
            int sz_out;
            float *curr_out;

            for (int i=0; i<layers.size(); i++){
                Module *layer = layers[i];

                sz_out = layer->sz_out;

                cudaMallocManaged(&curr_out, sz_out*sizeof(float));
                layer->forward(inp, curr_out);

                inp = curr_out;
            }

            // The last output is not freed here, it's used by loss
            // But we need to handle intermediate frees if we want to be memory efficient
            // The original code allocated curr_out for each layer and passed it as inp to next.
            // It didn't seem to free intermediate buffers in forward.
            // Wait, the original code had:
            // cudaMallocManaged(&curr_out, sizeof(float));
            // cudaFree(curr_out);
            // at the end of forward. That looks like a dummy alloc/free or a bug in original code.
            // We will stick to original logic for now but be aware of leaks.
        }

        void update(){
            for (int i=layers.size()-1; 0<=i; i--){
                Module *layer = layers[i];

                layer->update();
                layer->backward();
            }
        }
};

// ==========================================
// Training Loop
// ==========================================

void train_gpu(Sequential_GPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs){
    MSE_GPU mse(bs);

    int sz_inp = bs*n_in;
    float *cp_inp, *out;
    cudaMallocManaged(&cp_inp, sz_inp*sizeof(float));

    for (int i=0; i<n_epochs; i++){
        set_eq(cp_inp, inp, sz_inp);

        seq.forward(cp_inp, out);
        // seq.layers.back()->out is the output of the last layer
        mse.forward(seq.layers.back()->out, targ);

        mse.backward();
        seq.update();
        
        if (i % 10 == 0) {
             // Optional: Print loss every 10 epochs
             // To do this we need to run _forward which computes the loss value
             // But _forward modifies targ[sz_out] which is where loss is stored.
             // Let's just follow the original structure which prints at the end.
        }
    }

    seq.forward(inp, out);
    mse._forward(seq.layers.back()->out, targ);
    std::cout << "The final loss is: " << targ[bs] << std::endl;
}

// ==========================================
// Main
// ==========================================

int main(){
    std::chrono::steady_clock::time_point begin, end;

    int bs = 100000, n_in = 50, n_epochs = 100;
    int n_hidden = n_in/2;

    float *inp, *targ;
    cudaMallocManaged(&inp, bs*n_in*sizeof(float));
    cudaMallocManaged(&targ, (bs+1)*sizeof(float)); // +1 for loss storage

    begin = std::chrono::steady_clock::now();
    read_csv(inp, "../data/x.csv");
    read_csv(targ, "../data/y.csv");
    end = std::chrono::steady_clock::now();
    std::cout << "Data reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    Linear_GPU* lin1 = new Linear_GPU(bs, n_in, n_hidden);
    ReLU_GPU* relu1 = new ReLU_GPU(bs*n_hidden);
    Linear_GPU* lin2 = new Linear_GPU(bs, n_hidden, 1);

    std::vector<Module*> layers = {lin1, relu1, lin2};
    Sequential_GPU seq(layers);

    begin = std::chrono::steady_clock::now();
    train_gpu(seq, inp, targ, bs, n_in, n_epochs);
    end = std::chrono::steady_clock::now();
    std::cout << "Training time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    // Inference / Testing
    std::cout << "\nRunning Inference on first 5 samples..." << std::endl;
    // We can reuse the trained model 'seq' on the input data 'inp'
    // The output will be in seq.layers.back()->out
    
    // We need to run forward one last time to ensure 'out' is fresh if needed, 
    // but train_gpu already did a final forward pass.
    
    float* predictions = seq.layers.back()->out;
    
    for(int i=0; i<5; i++) {
        std::cout << "Sample " << i << ": Pred = " << predictions[i] << ", Target = " << targ[i] << std::endl;
    }

    return 0;
}