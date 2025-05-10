__global__ void naiveSoftmaxAttention(const float* Q,const float* K,const float* V, float* output, int M, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float sqrt_d = sqrtf(static_cast<float>(d));
    float* scores = new float[N];
    float* softmax_scores = new float[N];

    // Compute Q * K.T
    for (int col = 0; col < N; ++col) {
        float score = 0.0f;
        for (int k = 0; k < d; ++k) {
            score += Q[row * d + k] * K[col * d + k];
        }
        scores[col] = score / sqrt_d;
    }

    // Compute softmax of scores
    float max_score = scores[0];
    for (int i = 1; i < N; ++i) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < N; ++i) {
        softmax_scores[i] = expf(scores[i] - max_score);
        sum_exp += softmax_scores[i];
    }

    for (int i = 0; i < N; ++i) {
        softmax_scores[i] /= sum_exp;
    }

    // Compute softmax(Q * K.T) * V
    for (int col = 0; col < d; ++col) {
        float result = 0.0f;
        for (int k = 0; k < N; ++k) {
            result += softmax_scores[k] * V[k * d + col];
        }
        output[row * d + col] = result;
    }

    delete[] scores;
    delete[] softmax_scores;
}

// Wrapper function to launch the kernel
void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d)  {
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;

    naiveSoftmaxAttention<<<gridSize, blockSize>>>(Q, K, V, output, M, N, d);
}

