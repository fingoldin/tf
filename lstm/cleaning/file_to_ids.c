#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#define min(a, b)  ((a) < (b) ? (a) : (b))

typedef struct Node {
    struct Node *left;
    struct Node *right;
    char *word;
    int word_len;
    int id;
} node_t;

int nextid = 0;

int stringncmp(char *str1, char *str2, int n1, int n2)
{
    if(n1 < n2)
        return -1;
    else if(n1 > n2)
        return 1;

    for(int i = 0; i < n1; i++) {
        if(str1[i] < str2[i])
            return -1;
        else if(str1[i] > str2[i])
            return 1;
    }

    return 0;
}

node_t *createnode(char *word, int start, int end)
{
    node_t *n = malloc(sizeof(node_t));
    n->left = NULL;
    n->right = NULL;
    n->word = word + start;
    n->word_len = end - start;
    n->id = nextid++;
    
    return n;
}

int addword(node_t *node, char *word, int start, int end)
{
    int cmp = stringncmp(word + start, node->word, end - start, node->word_len);

    if(cmp == 0)
        return node->id;
    else if(cmp < 0) {
        if(node->left)
            return addword(node->left, word, start, end);
        else {
            node->left = createnode(word, start, end);
            return node->left->id;
        }
    }
    else {
        if(node->right)
            return addword(node->right, word, start, end);
        else {
            node->right = createnode(word, start, end);
            return node->right->id;
        }
    }
}

void output_nodes(node_t *node, FILE *file)
{
    for(int i = 0; i < node->word_len; i++)
        fputc(node->word[i], file);

    fprintf(file, "~%d\n", node->id);

    if(node->left)
        output_nodes(node->left, file);
    if(node->right)
        output_nodes(node->right, file);
}

void free_tree(node_t *node)
{
    if(node->left)
        free_tree(node->left);
    if(node->right)
        free_tree(node->right);

    free(node);
}

int main(int argc, char **argv)
{
    if(argc != 4) {
        printf("Please provide an input, id output, and word output file path, in that order\n");
        return 1;
    }

    FILE *input = fopen(argv[1], "r");
    if(!input) {
        printf("Could not open input file\n");
        return 2;
    }

    fseek(input, 0, SEEK_END);
    int input_length = ftell(input);
    fseek(input, 0, SEEK_SET);

    char *input_text = malloc(input_length);

    if(fread(input_text, 1, input_length, input) != input_length) {
        printf("Could not read input file\n");
        free(input_text);
        fclose(input);
        return 2;
    }

    fclose(input);

    int w_start = -1;

    int *word_ids = malloc(sizeof(int) * input_length / 2);
    int num_words = 0;

    node_t *root_n = NULL;

    for(int i = 0; i < input_length; i++) {
        char ch = input_text[i];
        
        if(isalpha(ch)) {
            if(ch == '-')
                printf("minus\n");
            if(w_start == -1)
                w_start = i;
        }
        else if(w_start != -1) {
            int id;

            switch(ch) {
            case ' ':
            case '\n':
                if(!root_n)
                    root_n = createnode(input_text, w_start, i);

                id = addword(root_n, input_text, w_start, i);
                word_ids[num_words++] = id;
                w_start = -1;
                break;
            case ',':
            case ';':
            case ':':
            case '.':
            case '!':
            case '?':
                if(!root_n)
                    root_n = createnode(input_text, w_start, i);

                id = addword(root_n, input_text, w_start, i); 
                word_ids[num_words++] = id;
                id = addword(root_n, input_text, i, i + 1);
                word_ids[num_words++] = id;
                w_start = -1;
                break; 
            }
        }
    }
  
    FILE *ids_output = fopen(argv[2], "w");
    if(!ids_output) {
        printf("Could not open ids output file\n");
        free(word_ids);
        free_tree(root_n);
        return 2;
    }
    
    output_nodes(root_n, ids_output);
    fclose(ids_output);

    free(input_text);
    free_tree(root_n);

    FILE *words_output = fopen(argv[3], "w");
    if(!words_output) {
        printf("Could not open words output file\n");
        free(word_ids);
        return 2;
    }
    
    for(int i = 0; i < (num_words - 1); i++)
        fprintf(words_output, "%d ", word_ids[i]);

    fprintf(words_output, "%d\n", word_ids[num_words - 1]);

    fclose(words_output);
    free(word_ids); 
    
    return 0;
}
