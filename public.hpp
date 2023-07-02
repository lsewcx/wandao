#include "image_preprocess.cpp"
#include "path_searching.cpp"
#include "ring_recognition.cpp"
#include "track_recognition.cpp"
#include "controlcenter_cal.cpp"


ImagePreprocess binarization;
PathSearching path1;
RingRecognition  ringRecognition;
TrackRecognition trackRecognition;
ControlCenterCal controlCenterCal;



enum RoadType
{
  BaseHandle = 0, // 基础赛道处理
  RingHandle,     // 环岛赛道处理
};
