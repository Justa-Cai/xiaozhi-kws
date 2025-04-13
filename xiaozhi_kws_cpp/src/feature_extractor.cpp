#include "feature_extractor.h"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>
#include <fftw3.h>
#include <cstring> // For memcpy
#include <stdexcept> // For runtime_error

// Helper to convert int16_t audio to double vector
std::vector<double> convert_audio_to_double(const int16_t* audio, size_t audio_len, bool normalize) {
    std::vector<double> audio_double(audio_len);
    double max_val = 0.0;
    if (normalize) {
        for (size_t i = 0; i < audio_len; ++i) {
            double val = static_cast<double>(audio[i]);
            audio_double[i] = val;
            if (std::abs(val) > max_val) {
                max_val = std::abs(val);
            }
        }
        if (max_val > 1.0) { // Normalize only if needed (int16 range is large)
             double scale = 32767.0;
             for (size_t i = 0; i < audio_len; ++i) {
                  audio_double[i] /= scale;
             }
        }
    } else {
         double scale = 32767.0;
         for (size_t i = 0; i < audio_len; ++i) {
             audio_double[i] = static_cast<double>(audio[i]) / scale; // Convert to [-1, 1] range anyway
         }
    }
    return audio_double;
}

// Constructor matching the header
FeatureExtractor::FeatureExtractor(const FeatureConfig& feature_config)
{
    initialize(feature_config); // Call the private initialize method
}

// Private initialize method (based on typical structure)
void FeatureExtractor::initialize(const FeatureConfig& config) {
    sample_rate_ = config.sample_rate;
    n_fft_ = config.n_fft;
    n_mels_ = config.n_mels;
    n_mfcc_ = config.n_mfcc;
    use_delta_ = config.use_delta;
    use_delta2_ = config.use_delta2;
    preemphasis_coeff_ = config.preemphasis_coeff;

    window_size_ = static_cast<int>(config.window_size_ms * sample_rate_ / 1000.0);
    window_stride_ = static_cast<int>(config.window_stride_ms * sample_rate_ / 1000.0);

    if (window_size_ <= 0 || window_stride_ <= 0) {
         throw std::runtime_error("Window size and stride must be positive.");
    }
     if (window_size_ > n_fft_) {
         std::cerr << "Warning: window_size (" << window_size_
                   << ") > n_fft (" << n_fft_ << "). Consider increasing n_fft." << std::endl;
     }

    // Call init functions declared in header
    init_window_function();
    init_mel_filterbank();
    init_dct_matrix();
    init_fft_plan();
}

// Destructor
FeatureExtractor::~FeatureExtractor() {
    if (fft_plan_ != nullptr) {
        fftw_destroy_plan(reinterpret_cast<fftw_plan>(fft_plan_));
    }
    // fftw_free is needed only if fftw_malloc was used directly
    // fft_input_ and fft_output_ are std::vectors, handled automatically
}

// Initialize Hann window (matches header declaration)
void FeatureExtractor::init_window_function() {
    window_func_.resize(window_size_);
    for (int i = 0; i < window_size_; ++i) {
        window_func_[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (window_size_ - 1)));
    }
}

// Hz to Mel conversion
double FeatureExtractor::hz_to_mel(double hz) {
    return 1127.0 * std::log(1.0 + hz / 700.0);
}

// Mel to Hz conversion
double FeatureExtractor::mel_to_hz(double mel) {
    return 700.0 * (std::exp(mel / 1127.0) - 1.0);
}

// Initialize Mel filterbank (matches header declaration)
void FeatureExtractor::init_mel_filterbank() {
    mel_filterbank_.resize(n_mels_, std::vector<double>(n_fft_ / 2 + 1, 0.0));

    double fmin = 0.0;
    double fmax = sample_rate_ / 2.0;
    double mel_fmin = hz_to_mel(fmin);
    double mel_fmax = hz_to_mel(fmax);
    std::vector<double> mel_points(n_mels_ + 2);
    for (int i = 0; i < n_mels_ + 2; ++i) {
        mel_points[i] = mel_fmin + (mel_fmax - mel_fmin) * i / (n_mels_ + 1);
    }

    std::vector<double> fft_freqs(n_fft_ / 2 + 1);
    for (int i = 0; i < n_fft_ / 2 + 1; ++i) {
        fft_freqs[i] = static_cast<double>(i * sample_rate_) / n_fft_;
    }

    for (int i = 0; i < n_mels_; ++i) {
        double left_mel = mel_points[i];
        double center_mel = mel_points[i + 1];
        double right_mel = mel_points[i + 2];

        double left_hz = mel_to_hz(left_mel);
        double center_hz = mel_to_hz(center_mel);
        double right_hz = mel_to_hz(right_mel);

        for (int j = 0; j < n_fft_ / 2 + 1; ++j) {
            if (fft_freqs[j] >= left_hz && fft_freqs[j] <= center_hz) {
                mel_filterbank_[i][j] = (fft_freqs[j] - left_hz) / (center_hz - left_hz);
            } else if (fft_freqs[j] > center_hz && fft_freqs[j] <= right_hz) {
                mel_filterbank_[i][j] = (right_hz - fft_freqs[j]) / (right_hz - center_hz);
            }
        }
    }
}

// Initialize DCT matrix (matches header declaration)
void FeatureExtractor::init_dct_matrix() {
    dct_matrix_.resize(n_mfcc_, std::vector<double>(n_mels_, 0.0));
    double scale = std::sqrt(2.0 / n_mels_);
    double scale0 = std::sqrt(1.0 / n_mels_);
    for (int k = 0; k < n_mfcc_; ++k) {
        for (int n = 0; n < n_mels_; ++n) {
            double factor = (k == 0) ? scale0 : scale;
            dct_matrix_[k][n] = factor * std::cos(M_PI * k * (2.0 * n + 1.0) / (2.0 * n_mels_));
        }
    }
}

// Initialize FFTW plan (matches header declaration)
void FeatureExtractor::init_fft_plan() {
    fft_input_.resize(n_fft_); // Resize the vector member
    fft_output_.resize(n_fft_ / 2 + 1); // Resize the vector member

    // Create FFTW plan for double precision R2C
    // The plan operates on the data pointers of the vectors
    fft_plan_ = fftw_plan_dft_r2c_1d(n_fft_,
                                     fft_input_.data(), // Use vector data pointer
                                     reinterpret_cast<fftw_complex*>(fft_output_.data()), // Cast vector data pointer
                                     FFTW_MEASURE); // Use MEASURE for potentially better performance

    // Pre-allocate other temporary vectors
    mel_energies_.resize(n_mels_);
    log_mel_energies_.resize(n_mels_);
    mfcc_coeffs_.resize(n_mfcc_);
    if (use_delta_) {
       delta_coeffs_.resize(n_mfcc_);
    }
}

// Apply preemphasis (private helper)
void FeatureExtractor::apply_preemphasis(const std::vector<double>& audio_in, std::vector<double>& audio_out) {
    if (preemphasis_coeff_ <= 0.0 || audio_in.empty()) {
        audio_out = audio_in; // Copy if no preemphasis
        return;
    }
    audio_out.resize(audio_in.size());
    audio_out[0] = audio_in[0];
    for (size_t i = 1; i < audio_in.size(); ++i) {
        audio_out[i] = audio_in[i] - preemphasis_coeff_ * audio_in[i - 1];
    }
}

// Apply window function to a frame (private helper)
void FeatureExtractor::apply_window(const std::vector<double>& frame, std::vector<double>& windowed_frame_out) {
    // Apply window to the input frame and store in fft_input_ for FFT
    if (frame.size() != window_size_) {
         throw std::runtime_error("Frame size does not match window size.");
    }
    for (int i = 0; i < window_size_; ++i) {
        fft_input_[i] = frame[i] * window_func_[i];
    }
    // Zero-pad the rest of fft_input_
    std::fill(fft_input_.begin() + window_size_, fft_input_.end(), 0.0);
    // windowed_frame_out is not actually used here as output goes to fft_input_
}

// Compute FFT for a windowed frame (operates on fft_input_)
void FeatureExtractor::compute_fft(const std::vector<double>& /* windowed_frame */, std::vector<std::complex<double>>& /* spectrum_out */) {
    // Input is assumed to be already prepared in fft_input_ by apply_window
    fftw_execute(reinterpret_cast<fftw_plan>(fft_plan_));
    // Output is automatically placed in fft_output_ vector by the plan
    // No need to copy, spectrum_out is not used here
}

// Compute Mel filterbank energies (private helper)
void FeatureExtractor::compute_mel_filterbank(const std::vector<std::complex<double>>& spectrum_in, std::vector<double>& mel_energies_out) {
    std::fill(mel_energies_out.begin(), mel_energies_out.end(), 0.0);
    for (int i = 0; i < n_mels_; ++i) {
        for (int j = 0; j < n_fft_ / 2 + 1; ++j) {
            double power_spec = std::norm(spectrum_in[j]); // Use spectrum_in (which is fft_output_)
            mel_energies_out[i] += mel_filterbank_[i][j] * power_spec;
        }
    }
}

// Compute log Mel energies (private helper)
void FeatureExtractor::compute_log_mel_energies(const std::vector<double>& mel_energies_in, std::vector<double>& log_mel_energies_out) {
    for (int i = 0; i < n_mels_; ++i) {
        log_mel_energies_out[i] = std::log(std::max(mel_energies_in[i], 1e-10));
    }
}

// Compute DCT to get MFCCs (private helper)
void FeatureExtractor::compute_dct(const std::vector<double>& log_mel_energies_in, std::vector<double>& mfcc_coeffs_out) {
    std::fill(mfcc_coeffs_out.begin(), mfcc_coeffs_out.end(), 0.0);
    for (int k = 0; k < n_mfcc_; ++k) {
        for (int n = 0; n < n_mels_; ++n) {
            mfcc_coeffs_out[k] += dct_matrix_[k][n] * log_mel_energies_in[n];
        }
    }
}

// Compute deltas (private helper, simplified width=3)
void FeatureExtractor::compute_deltas(const std::vector<std::vector<double>>& features_in, int delta_width, std::vector<std::vector<double>>& deltas_out) {
    if (features_in.empty() || delta_width <= 0) {
        deltas_out.clear();
        return;
    }
    int num_frames = features_in.size();
    int feature_dim = features_in[0].size();
    deltas_out.assign(num_frames, std::vector<double>(feature_dim, 0.0)); // Use assign for resize+fill

    if (delta_width != 3) {
         std::cerr << "Warning: C++ compute_deltas currently only implements width=3." << std::endl;
    }
    int half_width = 1; // for width 3
    double scale = 0.5; // Normalization factor for [-1, 0, 1] type weights

    for (int t = 0; t < num_frames; ++t) {
        for (int d = 0; d < feature_dim; ++d) {
            int prev_frame_idx = std::max(0, t - half_width);
            int next_frame_idx = std::min(num_frames - 1, t + half_width);
            deltas_out[t][d] = scale * (features_in[next_frame_idx][d] - features_in[prev_frame_idx][d]);
        }
    }
}


// Public Main feature extraction function matching header
std::vector<std::vector<float>> FeatureExtractor::extract_features(
        const int16_t* audio,
        size_t audio_len,
        bool normalize /*= false*/) // Match header default
{
    // 1. Convert input audio to double vector
    std::vector<double> audio_double = convert_audio_to_double(audio, audio_len, normalize);

    // 2. Apply preemphasis
    std::vector<double> audio_preemphasized;
    apply_preemphasis(audio_double, audio_preemphasized);

    int num_samples = audio_preemphasized.size();
    if (num_samples < window_size_) {
        // Pad if too short (to ensure at least one frame)
        int padding_needed = window_size_ - num_samples;
        audio_preemphasized.insert(audio_preemphasized.end(), padding_needed, 0.0);
        num_samples = audio_preemphasized.size();
    }

    // Calculate number of frames
    int num_frames = (num_samples >= window_size_) ? 1 + (num_samples - window_size_) / window_stride_ : 0;
    if (num_frames <= 0) {
         return {}; // Return empty if no frames can be computed
    }
    
    // --- MFCC Calculation Loop ---
    std::vector<std::vector<double>> mfcc_frames_double;
    mfcc_frames_double.reserve(num_frames);

    for (int i = 0; i < num_frames; ++i) {
        int start = i * window_stride_;

        // Extract frame (using audio_preemphasized)
        std::vector<double> frame(window_size_);
        int frame_len = std::min(window_size_, num_samples - start);
        if (frame_len < window_size_) {
            std::copy(audio_preemphasized.begin() + start, audio_preemphasized.begin() + start + frame_len, frame.begin());
            std::fill(frame.begin() + frame_len, frame.end(), 0.0);
        } else {
             std::copy(audio_preemphasized.begin() + start, audio_preemphasized.begin() + start + window_size_, frame.begin());
        }
       
        // Apply window (result goes to member fft_input_)
        apply_window(frame, mfcc_coeffs_); // Pass dummy vector

        // Compute FFT (operates on fft_input_, result in fft_output_)
        compute_fft(mfcc_coeffs_, fft_output_); // Pass dummy vectors

        // Compute Mel energies (using fft_output_, result in mel_energies_)
        compute_mel_filterbank(fft_output_, mel_energies_);

        // Compute log Mel energies (using mel_energies_, result in log_mel_energies_)
        compute_log_mel_energies(mel_energies_, log_mel_energies_);

        // Compute MFCCs (using log_mel_energies_, result in mfcc_coeffs_)
        compute_dct(log_mel_energies_, mfcc_coeffs_); // Overwrites mfcc_coeffs_
        mfcc_frames_double.push_back(mfcc_coeffs_); // Add the computed MFCC frame
    }
    // --- End MFCC Calculation Loop ---

    // --- Feature Combination --- 
    std::vector<std::vector<double>> final_features_double;

    // Compute delta features if requested (Only Delta1 supported currently)
    if (use_delta_) {
         std::vector<std::vector<double>> delta_frames_double;
         int delta_width = 3; // Match Python default
         compute_deltas(mfcc_frames_double, delta_width, delta_frames_double);

         // Concatenate MFCCs and Deltas
         final_features_double.reserve(num_frames);
         for (int i = 0; i < num_frames; ++i) {
             std::vector<double> combined_frame = mfcc_frames_double[i];
             if (i < delta_frames_double.size()) { 
                combined_frame.insert(combined_frame.end(), delta_frames_double[i].begin(), delta_frames_double[i].end());
             } else {
                 std::vector<double> zero_delta(n_mfcc_, 0.0);
                 combined_frame.insert(combined_frame.end(), zero_delta.begin(), zero_delta.end());
                 std::cerr << "Error: Delta frame size mismatch for index " << i << std::endl;
             }
             final_features_double.push_back(combined_frame);
         }
    } else {
        // If no delta, final features are just MFCCs
        final_features_double = mfcc_frames_double;
    }

    // Handle use_delta2_ (Currently unsupported in C++ delta calculation)
    if (use_delta2_) {
        std::cerr << "Warning: use_delta2 is true but not implemented in C++. Ignoring delta2 features." << std::endl;
        // Potentially compute and concatenate delta2 here if implemented
    }

    // 3. Convert final double features to float for return type
    std::vector<std::vector<float>> final_features_float(final_features_double.size());
    for(size_t i = 0; i < final_features_double.size(); ++i) {
        final_features_float[i].resize(final_features_double[i].size());
        for(size_t j = 0; j < final_features_double[i].size(); ++j) {
            final_features_float[i][j] = static_cast<float>(final_features_double[i][j]);
        }
    }

    return final_features_float;
}