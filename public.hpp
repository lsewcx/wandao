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
  CrossHandle,    // 十字道路处理
  FreezoneHandle, // 泛行区处理
  GarageHandle,   // 车库处理
  GranaryHandle,  // 粮仓处理
  DepotHandle,    // 修车厂处理
  BridgeHandle,   // 坡道(桥)处理
  SlowzoneHandle, // 慢行区（动物出没）处理
  FarmlandHandle, // 农田区域处理
};
