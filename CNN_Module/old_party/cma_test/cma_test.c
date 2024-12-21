#include <stdio.h>

#include "Thirdparty/cma/libxlnk_cma.h"

int main() {
    int len = 512;
    unsigned int *buf = (unsigned int*)cma_alloc(512, 0);

    int n = 512/sizeof(int);
    while(n--){
      printf("srcBuf:%x, srcBufPAddr:%x\n",buf, cma_get_phy_addr((void*)buf));
      *buf++;
    }
    

    cma_free((void*)buf);

    return 0;
}
