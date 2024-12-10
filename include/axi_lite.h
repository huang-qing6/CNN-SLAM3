#ifndef __AXI_LITE_H__
#define __AXI_LITE_H__

#include <iostream>
#include <sys/types.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// BASE ADDR 疑似256MB:0X00A0000000;4G:0X0400000000;224G:0X1000000000
// 假定现在BASE_ADDR 0X00A0000000,成功！
#define M_AXI_HPM0_FPD_BASE_ADDR 0X00A0000000

// REGISTER ADDR
// 寄存器怎么使用呢
//Write Data Reg Define
#define xPAA_WR_DATA_REG0    0
#define xPAA_WR_DATA_REG1    4
#define xPAA_WR_DATA_REG2    8
#define xPAA_WR_DATA_REG3    12
#define xPAA_WR_DATA_REG4    16
#define xPAA_WR_DATA_REG5    20
#define xPAA_WR_DATA_REG6    24
#define xPAA_WR_DATA_REG7    28
//Write Data Control Reg
#define xPAA_WR_CTRL_REG     32
//Read Data Reg Define
#define xPAA_RD_DATA_REG0    36
#define xPAA_RD_DATA_REG1    40
#define xPAA_RD_DATA_REG2    44
#define xPAA_RD_DATA_REG3    48
#define xPAA_RD_DATA_REG4    52
#define xPAA_RD_DATA_REG5    56
#define xPAA_RD_DATA_REG6    60
#define xPAA_RD_DATA_REG7    64
//Read Data Control Reg
#define xPAA_RD_CTRL_REG     68


// func
class AXILITE{
    public:
        AXILITE();

        void send(unsigned int dataAddr, unsigned int dataLength);
        //void recv(unsigned int bufAddr, unsigned int dataLength);

        void destroy();
    private:
        unsigned int* axi_hpm_addr;

        void setReg(unsigned int regOffset, unsigned int value);

        unsigned int getReg(unsigned int regOffset);   
};


#endif