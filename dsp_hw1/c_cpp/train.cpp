#include "hmm.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define pb  push_back
#define a   (model->transition)
#define b   (model->observation)
#define pi  (model->initial)

using namespace std;

static double alpha[MAX_SEQ][MAX_STATE];
static double beta[MAX_SEQ][MAX_STATE];

static double gamma_sum_by_obsv[MAX_STATE][MAX_OBSERV];
static double gamma_at_1[MAX_STATE];
static double gamma_at_t[MAX_STATE];
static double epsilon_sum[MAX_STATE][MAX_STATE];


static vector<string> load_seq(char *seq_file) {
    vector<string> ret;
    FILE *fp = fopen(seq_file, "r");
    char l[MAX_LINE];
    while(fscanf(fp, "%s", l) > 0)
        ret.pb(string(l));
    fclose(fp);
    return ret;
}

static void forward(HMM *model, string& o) {
    int N = model->state_num;
    int T = (int)o.size();
    // Initialization
    for(int i = 0; i < N; i++)
        alpha[0][i] = pi[i] * b[o[0] - 'A'][i];
    // Induction
    for(int t = 0; t < T - 1; t++) {
        for(int i = 0; i < N; i++) {
            double sum = 0;
            for(int k = 0; k < N; k++)
                sum += alpha[t][k] * a[k][i];
            alpha[t+1][i] = sum * b[o[t+1] - 'A'][i];
        }
    }
    return;
}

static void backward(HMM *model, string& o) {
    int N = model->state_num;
    int T = (int)o.size();
    // Initialization
    for(int i = 0; i < N; i++)
        beta[T-1][i] = 1;
    // Induction
    for(int t = T-2; t >= 0; t--) {
        for(int i = 0; i < N; i++) {
            beta[t][i] = 0;
            for(int j = 0; j < N; j++) {
                beta[t][i] += a[i][j] * b[o[t+1] - 'A'][j] * beta[t+1][j];
            }
        }
    }
    return;
}

static void cal_gamma(HMM *model, string& o) {
    int N = model->state_num;
    int T = (int)o.size();
    for(int t = 0; t < T; t++) {
        double sum = 0;
        for(int i = 0; i < N; i++) sum += alpha[t][i] * beta[t][i];
        for(int i = 0; i < N; i++) {
            double gamma = alpha[t][i] * beta[t][i] / sum;
            if (t == 0)     gamma_at_1[i] += gamma;
            if (t == T - 1) gamma_at_t[i] += gamma;
            gamma_sum_by_obsv[i][o[t] - 'A'] += gamma;
        }
    }
    return;
}

static void cal_epsilon(HMM *model, string& o) {
    int N = model->state_num;
    int T = (int)o.size();
    for(int t = 0; t < T - 1; t++) {
        double sum = 0;
        double epsilon[MAX_STATE][MAX_STATE];
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                epsilon[i][j] = alpha[t][i] * a[i][j] * b[o[t+1] - 'A'][j] * beta[t+1][j];
                sum += epsilon[i][j];
            }
        }
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
                epsilon_sum[i][j] += (epsilon[i][j] / sum);
    }
    return;
}

static void init_variables() {
    memset(alpha, 0, sizeof(alpha));
    memset(beta, 0, sizeof(beta));
    memset(gamma_sum_by_obsv, 0, sizeof(gamma_sum_by_obsv));
    memset(gamma_at_1, 0, sizeof(gamma_at_1));
    memset(gamma_at_t, 0, sizeof(gamma_at_t));
    memset(epsilon_sum, 0, sizeof(epsilon_sum));
    return;
}


static void update_params(HMM *model, int seq_size) {
    int N = model->state_num;
    // Update pi
    for(int i = 0; i < N; i++)
        pi[i] = gamma_at_1[i] / seq_size;
    // Update a
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double sum = 0;
            for(double k: gamma_sum_by_obsv[i]) sum += k;
            a[i][j] = epsilon_sum[i][j] / (sum - gamma_at_t[i]);
        }
    }
    // Update b
    for(int i = 0; i < N; i++)
        for(int j = 0; j < model->observ_num; j++) {
            double sum = 0;
            for(double k: gamma_sum_by_obsv[i]) sum += k;
            b[j][i] = gamma_sum_by_obsv[i][j] / sum;
        }
    return;
}

static void train(HMM *model, vector<string> seq, int iteration) {
    for(int _i = 0; _i < iteration; _i++) {
        init_variables();
        for(string o: seq) {
            forward(model, o);
            backward(model, o);
            cal_gamma(model, o);
            cal_epsilon(model, o);
        }
        update_params(model, int(seq.size()));
    }
    return;
}

int main(int argc, char **argv) {
    // parse arguments
    if (argc != 5) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "$./train iteration model_init.txt seq_model_01.txt model_01.txt\n");        
        exit(1);
    }
    int iteration = atoi(argv[1]);
    // load initial model
    HMM model;
    loadHMM(&model, argv[2]);
    // load sequences
    vector<string> seq = load_seq(argv[3]);
    // start training
    train(&model, seq, iteration);
    // dump results
    FILE *fp = fopen(argv[4], "w");
    dumpHMM(fp, &model);
    fclose(fp);
    return 0;
}