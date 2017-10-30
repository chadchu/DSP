#include "hmm.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

#define MAX_NUM	10
#define pb  	push_back
#define a   	(model->transition)
#define b   	(model->observation)
#define pi  	(model->initial)

using namespace std;

static double delta[MAX_SEQ][MAX_STATE];

struct result {
	int idx;
	double prob;
};

static vector<string> load_seq(char *seq_file) {
    vector<string> ret;
    FILE *fp = fopen(seq_file, "r");
    char l[MAX_LINE];
    while(fscanf(fp, "%s", l) > 0)
        ret.pb(string(l));
    fclose(fp);
    return ret;
}

static double viterbi(HMM *model, string o) {
	int N = model->state_num;
    int T = (int)o.size();
    // Initialization
    for(int i = 0; i < N; i++)
        delta[0][i] = pi[i] * b[o[0] - 'A'][i];
    // Recursion
    for(int t = 0; t < T - 1; t++)
    	for(int j = 0; j < N; j++) {
    		double _max = -1.0;
    		for(int i = 0; i < N; i++) {
    			double _ = delta[t][i]*a[i][j];
    			_max = _ > _max? _ : _max;
    		}
	    	delta[t+1][j] = _max * b[o[t+1] - 'A'][j];
    	}
    // return max probability
    return *max_element(delta[T-1], delta[T-1] + N);
}

static result compare_models(HMM *models, int n_models, string o) {
	double max_prob = -1;
	int max_idx;
	for(int i = 0; i < n_models; i++) {
		double prob = viterbi(&models[i], o);
		if (prob > max_prob) {
			max_prob = prob;
			max_idx = i;
		}
	}
	return (result){max_idx, max_prob};
}


int main(int argc, char **argv) {
	if (argc != 4) {
		fprintf(stderr, "Usage:\n");
		fprintf(stderr, "./test  modellist.txt  testing_data.txt  result.txt\n");
		exit(1);
	}
	// load model
	HMM models[MAX_NUM];
	int n_models = load_models(argv[1], models, MAX_NUM);
	// load test data
	vector<string> test_data = load_seq(argv[2]);
	// TEST
	FILE *fp = fopen(argv[3], "w");
	for(string s: test_data) {
		result res = compare_models(models, n_models, s);
		fprintf(fp, "%s %e\n", models[res.idx].model_name, res.prob);
	}
	fclose(fp);
	return 0;
}