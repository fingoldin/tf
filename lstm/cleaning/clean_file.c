#include <stdio.h>
#include <ctype.h>

int main(int argc, char **argv)
{
    if(argc != 3) {
        printf("Please provide an input and ouput file, in that order\n");
        return 1;
    }

    FILE *input = fopen(argv[1], "r");
    if(!input) {
        printf("Could not open input file\n");
        return 2;
    }

    FILE *output = fopen(argv[2], "w");
    if(!output) {
        printf("Could not open output file\n");
        return 2;
    }
    
    char *special = " \n,;:.!?-";
    
    char c;
    while((c = fgetc(input)) != EOF) {
        if(isalpha(c))
            fputc(tolower(c), output);
        else {
            for(int i = 0; special[i] != '\0'; i++) {
                if(c == special[i]) {
                    fputc(c, output);
                    break;
                }
            }
        }
    }

    fclose(input);
    fclose(output);

    return 0;
}

