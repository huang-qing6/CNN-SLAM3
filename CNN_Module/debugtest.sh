#数据集版
if [ "$1" == "mono" ]; then
    ./mono_euroc ../Vocabulary/ORBvoc.bin ./EuRoC.yaml ../../dataset/MH01 ../../EuRoC_TimeStamps/MH01.txt
elif [ "$1" == "monodebug" ]; then
    gdb --args ./mono_euroc ../Vocabulary/ORBvoc.bin ./EuRoC.yaml ../../dataset/MH01 ../../EuRoC_TimeStamps/MH01.txt
elif [ "$1" == "rgbd" ]; then
    ./rgbd ../Vocabulary/ORBvoc.bin ./TUM1.yaml ../../dataset/rgbd ../../dataset/rgbd/associate.txt 
elif [ "$1" == "rgbddebug" ]; then
    gdb --args ./rgbd ../Vocabulary/ORBvoc.bin ./TUM1.yaml ../../dataset/rgbd ../../dataset/rgbd/associate.txt 
fi

#摄像头版
# gdb --args ./mono_euroc ../Vocabulary/ORBvoc.bin ./EuRoC.yaml
