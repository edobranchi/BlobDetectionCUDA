#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

//controllo degli errori di CUDA
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore CUDA: %s - %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

// Kernel CUDA per trovare le differenze tra immagini blurrate con Gaussian blur
__global__ void findDifferenceKernel(float* image1, float* image2, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;  // Guardo se esco dai bordi dell'immagine

    int idx = y * width + x;  // Appiattisco in 1D
    output[idx] = image2[idx] - image1[idx];  // Faccio differenza
}

// Kernel CUDA per trovare gli estremi locali (massimi e minimi)
__global__ void findExtremaKernel(float* dogPrev, float* dogCurrent, float* dogNext,
                                  int width, int height, float* outputMaxima, float* outputMinima) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;  // Guardo se esco fuori

    float value = dogCurrent[y * width + x];  // Valore del pixel attuale
    bool isMax = true;
    bool isMin = true;

    // Controll0 tutti i 26 vicini nel cubo 3x3x3
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dz == 0 && dy == 0 && dx == 0) continue;

                int nx = x + dx;
                int ny = y + dy;
                float neighbor;

                if (dz == -1) {
                    neighbor = dogPrev[ny * width + nx];
                } else if (dz == 0) {
                    neighbor = dogCurrent[ny * width + nx];
                } else {
                    neighbor = dogNext[ny * width + nx];
                }

                if (value <= neighbor) isMax = false;
                if (value >= neighbor) isMin = false;
            }
        }
    }

    outputMaxima[y * width + x] = isMax ? value : 0.0f;  // Salvo massimi/minimi
    outputMinima[y * width + x] = isMin ? value : 0.0f;
}

//rilevo i blob usando DoG, con gaussianblur calcolato sulla CPU
vector<KeyPoint> detectBlobsCUDA(const Mat& image, double sigma, int numScales, double k) {
    int width = image.cols;  // Dimensioni dell'immagine
    int height = image.rows;
    size_t imageSize = width * height * sizeof(float);  // Dimensione dell'immagine in byte

    // Creazione della piramide di immagini blurrate su CPU
    vector<Mat> gaussianPyramid;
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < numScales; i++) {
        double currentSigma = sigma * pow(k, i);
        Mat blurred;
        GaussianBlur(image, blurred, Size(0, 0), currentSigma);  // Uso GaussianBlur di OpenCV
        gaussianPyramid.push_back(blurred);
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedBlur = end - start;
    std::cout << "Piramide Gaussiana calcolata in: " << elapsedBlur.count() << " secondi" << std::endl;

    // Allocazione della memoria sulla GPU e copia delle immagini blurrate
    vector<float*> d_gaussian(numScales);  // Puntatori alle immagini blurrate sulla GPU
    for (int i = 0; i < numScales; i++) {
        cudaMalloc(&d_gaussian[i], imageSize);  // Alloco memoria sulla GPU
        cudaMemcpy(d_gaussian[i], gaussianPyramid[i].ptr<float>(), imageSize, cudaMemcpyHostToDevice);  // Copio i dati sulla GPU
    }

    // Impostazione delle dimensioni dei blocchi e delle griglie
    dim3 blockSize(16, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);  // Dimensioni della griglia calcolate sulla base delle dimensioni dell'immagine

    // Creo dello spazio per le immagini di differenza (DoG)
    vector<float*> d_dog(numScales - 1);  // Puntatori alle immagini DoG sulla GPU
    for (int i = 0; i < numScales - 1; i++) {
        cudaMalloc(&d_dog[i], imageSize);  // Alloco memoria sulla GPU
    }

    // Calcolo della Differenza di Gaussiane (DoG) sulla GPU
    for (int i = 0; i < numScales - 1; i++) {
        findDifferenceKernel<<<gridSize, blockSize>>>(d_gaussian[i], d_gaussian[i + 1], d_dog[i], width, height);  // Chiamo del kernel
    }

    // Alloco memoria per il rilevamento degli estremi
    float *d_maxima, *d_minima;  // Puntatori alle immagini di massimi e minimi sulla GPU
    cudaMalloc(&d_maxima, imageSize);
    cudaMalloc(&d_minima, imageSize);

    vector<KeyPoint> keypoints;

    // Trova gli estremi in ogni livello DoG
    for (int i = 1; i < numScales - 2; i++) {
        cudaMemset(d_maxima, 0, imageSize);  // libera la memoria dalle immagini di massimi e minimi
        cudaMemset(d_minima, 0, imageSize);

        //trovo gli estremi locali
        findExtremaKernel<<<gridSize, blockSize>>>(d_dog[i - 1], d_dog[i], d_dog[i + 1],
                                                   width, height, d_maxima, d_minima);
        cudaDeviceSynchronize();  // Aspetto che il kernel finisca

        // Riporto i risultati sulla CPU
        Mat maxima(height, width, CV_32F);
        Mat minima(height, width, CV_32F);
        cudaMemcpy(maxima.ptr<float>(), d_maxima, imageSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(minima.ptr<float>(), d_minima, imageSize, cudaMemcpyDeviceToHost);

        // Estraggo i keypoint
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                if (maxima.at<float>(y, x) != 0 || minima.at<float>(y, x) != 0) {
                    keypoints.push_back(KeyPoint(Point2f(x, y), sigma * pow(k, i)));
                }
            }
        }
    }

    // Libero memoria sulla GPU
    for (auto ptr : d_gaussian) cudaFree(ptr);  //immagini Blurrate
    for (auto ptr : d_dog) cudaFree(ptr);  // immagini DoG
    cudaFree(d_maxima);
    cudaFree(d_minima);

    return keypoints;
}

int main() {
    // Carico l'immagine
    Mat image = imread("/data01/pc24edobra/test_images/paesaggio-grande.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Errore durante il caricamento dell'immagine" << endl;
        return -1;
    }
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    // Parametri
    double sigma = 3;
    int numScales = 5;
    double k = 2;

    // Main computation
    auto start = chrono::high_resolution_clock::now();
    vector<KeyPoint> keypoints = detectBlobsCUDA(image, sigma, numScales, k);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;
    cout << "Tempo di esecuzione CUDA: " << elapsed.count() << " secondi" << endl;
    cout << "Trovati " << keypoints.size() << " keypoints" << endl;

    //Commentare per non salvare l'immagine
    //disegno i keypoints
    image.convertTo(image, CV_8UC1, 255.0);
    Mat output;
    drawKeypoints(image, keypoints, output, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    if (output.empty()) {
        cerr << "Errore: Impossibile disegnare i keypoints." << endl;
        return -1;
    }
    //salvo l'immagine di output
    if (!imwrite("/data01/pc24edobra/Desktop/test_output/paesaggio-grande-out-cuda.jpg", output)) {
        cerr << "Errore: Impossibile salvare l'immagine di output." << endl;
        return -1;
    }

    return 0;
}
