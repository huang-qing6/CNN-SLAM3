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

#define CTRL_BUS_ADDR_AP_CTRL            0x000
#define CTRL_BUS_ADDR_GIE                0x004
#define CTRL_BUS_ADDR_IER                0x008
#define CTRL_BUS_ADDR_ISR                0x00c
#define CTRL_BUS_ADDR_INPUT_R_DATA       0x010
#define CTRL_BUS_BITS_INPUT_R_DATA       32
#define CTRL_BUS_ADDR_INPUT1_DATA        0x018
#define CTRL_BUS_BITS_INPUT1_DATA        32
#define CTRL_BUS_ADDR_INPUT2_DATA        0x020
#define CTRL_BUS_BITS_INPUT2_DATA        32
#define CTRL_BUS_ADDR_INPUT3_DATA        0x028
#define CTRL_BUS_BITS_INPUT3_DATA        32
#define CTRL_BUS_ADDR_OUTPUT_R_DATA      0x030
#define CTRL_BUS_BITS_OUTPUT_R_DATA      32
#define CTRL_BUS_ADDR_OUTPUT1_DATA       0x038
#define CTRL_BUS_BITS_OUTPUT1_DATA       32
#define CTRL_BUS_ADDR_WEIGHT_DATA        0x040
#define CTRL_BUS_BITS_WEIGHT_DATA        32
#define CTRL_BUS_ADDR_BETA_DATA          0x048
#define CTRL_BUS_BITS_BETA_DATA          32
#define CTRL_BUS_ADDR_INFM_NUM_DATA      0x050
#define CTRL_BUS_BITS_INFM_NUM_DATA      32
#define CTRL_BUS_ADDR_OUTFM_NUM_DATA     0x058
#define CTRL_BUS_BITS_OUTFM_NUM_DATA     32
#define CTRL_BUS_ADDR_KERNEL_SIZE_DATA   0x060
#define CTRL_BUS_BITS_KERNEL_SIZE_DATA   32
#define CTRL_BUS_ADDR_KERNEL_STRIDE_DATA 0x068
#define CTRL_BUS_BITS_KERNEL_STRIDE_DATA 32
#define CTRL_BUS_ADDR_INPUT_W_DATA       0x070
#define CTRL_BUS_BITS_INPUT_W_DATA       32
#define CTRL_BUS_ADDR_INPUT_H_DATA       0x078
#define CTRL_BUS_BITS_INPUT_H_DATA       32
#define CTRL_BUS_ADDR_OUTPUT_W_DATA      0x080
#define CTRL_BUS_BITS_OUTPUT_W_DATA      32
#define CTRL_BUS_ADDR_OUTPUT_H_DATA      0x088
#define CTRL_BUS_BITS_OUTPUT_H_DATA      32
#define CTRL_BUS_ADDR_PADDING_DATA       0x090
#define CTRL_BUS_BITS_PADDING_DATA       32
#define CTRL_BUS_ADDR_ISNL_DATA          0x098
#define CTRL_BUS_BITS_ISNL_DATA          1
#define CTRL_BUS_ADDR_ISBN_DATA          0x0a0
#define CTRL_BUS_BITS_ISBN_DATA          1
#define CTRL_BUS_ADDR_TM_DATA            0x0a8
#define CTRL_BUS_BITS_TM_DATA            32
#define CTRL_BUS_ADDR_TN_DATA            0x0b0
#define CTRL_BUS_BITS_TN_DATA            32
#define CTRL_BUS_ADDR_TR_DATA            0x0b8
#define CTRL_BUS_BITS_TR_DATA            32
#define CTRL_BUS_ADDR_TC_DATA            0x0c0
#define CTRL_BUS_BITS_TC_DATA            32
#define CTRL_BUS_ADDR_MLOOPS_DATA        0x0c8
#define CTRL_BUS_BITS_MLOOPS_DATA        32
#define CTRL_BUS_ADDR_NLOOPS_DATA        0x0d0
#define CTRL_BUS_BITS_NLOOPS_DATA        32
#define CTRL_BUS_ADDR_RLOOPS_DATA        0x0d8
#define CTRL_BUS_BITS_RLOOPS_DATA        32
#define CTRL_BUS_ADDR_CLOOPS_DATA        0x0e0
#define CTRL_BUS_BITS_CLOOPS_DATA        32
#define CTRL_BUS_ADDR_LAYERTYPE_DATA     0x0e8
#define CTRL_BUS_BITS_LAYERTYPE_DATA     32
#define CTRL_BUS_ADDR_INPUTQ_DATA        0x0f0
#define CTRL_BUS_BITS_INPUTQ_DATA        32
#define CTRL_BUS_ADDR_OUTPUTQ_DATA       0x0f8
#define CTRL_BUS_BITS_OUTPUTQ_DATA       32
#define CTRL_BUS_ADDR_WEIGHTQ_DATA       0x100
#define CTRL_BUS_BITS_WEIGHTQ_DATA       32
#define CTRL_BUS_ADDR_BETAQ_DATA         0x108
#define CTRL_BUS_BITS_BETAQ_DATA         32
#define CTRL_BUS_ADDR_TROW_LOOPS_DATA    0x110
#define CTRL_BUS_BITS_TROW_LOOPS_DATA    32

#define WriteReg(BaseAddress, RegOffset, Data) *(volatile unsigned int*)((BaseAddress) + (RegOffset)) = (Data)
#define ReadReg(BaseAddress, RegOffset) *(volatile unsigned int*)((BaseAddress) + (RegOffset))

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