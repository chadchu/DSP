#include <stdio.h>
#include <stdlib.h>
#include "File.h"
#include "LM.h"
#include "Ngram.h"
#include "Vocab.h"
#include "VocabMap.h"

#define MAX_LENGTH      200
#define MAX_NEXT        1000

int main(int argc, char **argv){
    if (argc != 9) {
        fprintf(stderr, "Usage:\n./mydisambig -text [text_file] -map [map_file] -lm [lm_file] -order [order]\n");
        exit(1);
    }
    // parsing arguments
    File textFile(argv[2], "r");
    File mapFile(argv[4], "r");
    File lmFile(argv[6], "r");
    int order = atoi(argv[8]);

    // load mapping
    Vocab zhuyin, big5;
    VocabMap mapping(zhuyin, big5);
    mapping.read(mapFile);
    mapFile.close();

    // load language model
    Vocab voc;
    Ngram lm(voc, order);
    lm.read(lmFile);
    lmFile.close();

    // loading text data
    char *line;
    while (line = textFile.getline()) {

        VocabString words[MAX_LENGTH];
        unsigned int len = Vocab::parseWords(line, &(words[1]), MAX_LENGTH);
        words[0] = "<s>";
        words[len + 1] = "</s>";
        len += 2;
        
        VocabIndex INDEX[MAX_LENGTH][MAX_NEXT] = {0};
        LogP PROB[MAX_LENGTH][MAX_NEXT] = {{0.0}};
        int backward[MAX_LENGTH][MAX_NEXT] = {{0}};
        int idx_num[MAX_LENGTH] = {0};

        // initialization
        VocabIndex tmp = voc.getIndex(words[0]);
        PROB[0][0] = 1.0;
        INDEX[0][0] = big5.getIndex(words[0]);
        idx_num[0] = 1;

        // forward
        for (int i = 1; i < len; i++) {
            int n = 0;
            VocabMapIter it(mapping, zhuyin.getIndex(words[i]));
            it.init();
            Prob p;
            VocabIndex idx;
            while (it.next(idx, p)) {
                tmp = voc.getIndex(big5.getWord(idx));
                if (tmp == Vocab_None) tmp = voc.getIndex(Vocab_Unknown);
                
                // find max probability of previous column
                LogP max_prob = LogP_Zero;
                for (int j = 0; j < idx_num[i-1]; j++) {
                    VocabIndex prev = voc.getIndex(big5.getWord(INDEX[i - 1][j]));
                    if (prev == Vocab_None) prev = voc.getIndex(Vocab_Unknown);
                    VocabIndex context[] = {prev, Vocab_None};
                    LogP P = lm.wordProb(tmp, context);
                    if (P == LogP_Zero) P = -1000;
                    P += PROB[i - 1][j];
                    if (P > max_prob) {
                        max_prob = P;
                        backward[i][n] = j;
                    }
                }

                PROB[i][n] = max_prob;
                INDEX[i][n] = idx;
                n++;

            }
            idx_num[i] = n;
        }

        // find maximum probability
        LogP max_prob = LogP_Zero;
        int loc;
        for (int i = 0 ; i < idx_num[len - 1]; i++) {
            if (PROB[len - 1][i] > max_prob) {
                max_prob = PROB[len - 1][i];
                loc = i;
            }
        }

        // backward
        VocabString path[MAX_LENGTH];
        for (int i = len - 1; i >= 0; i--) {
            path[i] = big5.getWord(INDEX[i][loc]);
            loc = backward[i][loc];
        }
        
        for (int i = 0; i < len; i++)
            printf("%s%c", path[i], " \n"[i == len - 1]);


    }

} 
