#include "axi_lite.h"

// 初始化
AXILITE::AXILITE(){
    unsigned int baseaddr = M_AXI_HPM0_FPD_BASE_ADDR;

    int aximem = open("/dev/mem", O_RDWR | O_SYNC);
    if(aximem == -1)
        printf("err axilite open\n");

    // 申请axi_gpm_fpd映射
    axi_hpm_addr = (unsigned int*)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, aximem, baseaddr);
    if(axi_hpm_addr == MAP_FAILED)
        printf("err mmap\n");

    if(close(aximem))
        printf("err close aximem\n");

}

void AXILITE::send(unsigned int dataAddr, unsigned int dataLength){
    setReg(dataAddr, dataLength);
}

void AXILITE::destroy(){
    munmap(axi_hpm_addr, 4096);
}


// 不需要volatile?
void AXILITE::setReg(unsigned int regOffset, unsigned int value){
    axi_hpm_addr[regOffset >> 2] = value;
}


unsigned int AXILITE::getReg(unsigned int regOffset){
    return 0;
}
