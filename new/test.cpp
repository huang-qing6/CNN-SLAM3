#include <iostream>

#include "axi_lite.h"

using namespace std;

int main(){
    AXILITE A;
    A.send(0X04, 1);


    return 0;
}