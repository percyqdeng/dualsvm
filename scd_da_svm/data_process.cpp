#include "data_process.h"


int libsvm_load_data(char *filename)
// loads the same format as LIBSVM
{
    int index; double value;
    int elements, i;
    FILE *fp = fopen(filename,"r");
    lasvm_sparsevector_t* v;

    if(fp == NULL)
    {
        fprintf(stderr,"Can't open input file \"%s\"\n",filename);
        exit(1);
    }
    else
        printf("loading \"%s\"..  \n",filename);
    int splitpos=0;

    int msz = 0;
    elements = 0;
    while(1)
    {
        int c = fgetc(fp);
        switch(c)
        {
        case '\n':
            if(splits.size()>0)
            {
                if(splitpos<(int)splits.size() && splits[splitpos].x==msz)
                {
                    v=lasvm_sparsevector_create();
                    X.push_back(v);	splitpos++;
                }
            }
            else
            {
                v=lasvm_sparsevector_create();
                X.push_back(v);
            }
            ++msz;
            //printf("%d\n",m);
            elements=0;
            break;
        case ':':
            ++elements;
            break;
        case EOF:
            goto out;
        default:
            ;
        }
    }
 out:
    rewind(fp);


    max_index = 0;splitpos=0;
    for(i=0;i<msz;i++)
    {

        int write=0;
        if(splits.size()>0)
        {
            if(splitpos<(int)splits.size() && splits[splitpos].x==i)
            {
                write=2;splitpos++;
            }
        }
        else
            write=1;

        int label;
        fscanf(fp,"%d",&label);
        //	printf("%d %d\n",i,label);
        if(write)
        {
            if(splits.size()>0)
            {
                if(splits[splitpos-1].y!=0)
                    Y.push_back(splits[splitpos-1].y);
                else
                    Y.push_back(label);
            }
            else
                Y.push_back(label);
        }

        while(1)
        {
            int c;
            do {
                c = getc(fp);
                if(c=='\n') goto out2;
            } while(isspace(c));
            ungetc(c,fp);
            fscanf(fp,"%d:%lf",&index,&value);

            if (write==1) lasvm_sparsevector_set(X[m+i],index,value);
            if (write==2) lasvm_sparsevector_set(X[splitpos-1],index,value);
            if (index>max_index) max_index=index;
        }
    out2:
        label=1; // dummy
    }

    fclose(fp);

    msz=X.size()-m;
    printf("examples: %d   features: %d\n",msz,max_index);

    return msz;
}

void load_data_file(char *filename)
{
    int msz,i,ft;
    splits.resize(0);

    int bin=binary_files;
    if(bin==0) // if ascii, check if it isn't a split file..
    {
        FILE *f=fopen(filename,"r");
        if(f == NULL)
        {
            fprintf(stderr,"Can't open input file \"%s\"\n",filename);
            exit(1);
        }
        char c; fscanf(f,"%c",&c);
        if(c=='f') bin=2; // found split file!
    }

    switch(bin)  // load diferent file formats
    {
    case 0: // libsvm format
        msz=libsvm_load_data(filename); break;
    case 1:
        msz=binary_load_data(filename); break;
    case 2:
        ft=split_file_load(filename);
        if(ft==0)
        {msz=libsvm_load_data(filename); break;}
        else
        {msz=binary_load_data(filename); break;}
    default:
        fprintf(stderr,"Illegal file type '-B %d'\n",bin);
        exit(1);
    }

    if(kernel_type==RBF)
    {
        x_square.resize(m+msz);
        for(i=0;i<msz;i++)
            x_square[i+m]=lasvm_sparsevector_dot_product(X[i+m],X[i+m]);
    }

    if(kgamma==-1)
        kgamma=1.0/ ((double) max_index); // same default as LIBSVM

    m+=msz;
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
    int i; int clss; double weight;

    // parse options
    for(i=1;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        ++i;
        switch(argv[i-1][1])
        {
        case 'o':
            optimizer = atoi(argv[i]);
            break;
        case 't':
            kernel_type = atoi(argv[i]);
            break;
        case 's':
            selection_type = atoi(argv[i]);
            break;
        case 'l':
            while(1)
            {
                select_size.push_back(atof(argv[i]));
                ++i; if((argv[i][0]<'0') || (argv[i][0]>'9')) break;
            }
            i--;
            break;
        case 'd':
            degree = atof(argv[i]);
            break;
        case 'g':
            kgamma = atof(argv[i]);
            break;
        case 'r':
            coef0 = atof(argv[i]);
            break;
        case 'm':
            cache_size = (int) atof(argv[i]);
            break;
        case 'c':
            C = atof(argv[i]);
            break;
        case 'w':
            clss= atoi(&argv[i-1][2]);
            weight = atof(argv[i]);
            if (clss>=1) C_pos=weight; else C_neg=weight;
            break;
        case 'b':
            use_b0=atoi(argv[i]);
            break;
        case 'B':
            binary_files=atoi(argv[i]);
            break;
        case 'e':
            epsgr = atof(argv[i]);
            break;
        case 'p':
            epochs = atoi(argv[i]);
            break;
        case 'D':
            deltamax = atoi(argv[i]);
            break;
        case 'C':
            candidates = atoi(argv[i]);
            break;
        case 'T':
            termination_type = atoi(argv[i]);
            break;
        default:
            fprintf(stderr,"unknown option\n");
            exit_with_help();
        }
    }

    saves=select_size.size();
    if(saves==0) select_size.push_back(100000000);

    // determine filenames

    if(i>=argc)
        exit_with_help();

    strcpy(input_file_name, argv[i]);

    if(i<argc-1)
        strcpy(model_file_name,argv[i+1]);
    else
    {
        char *p = strrchr(argv[i],'/');
        if(p==NULL)
            p = argv[i];
        else
            ++p;
        sprintf(model_file_name,"%s.model",p);
    }

}

