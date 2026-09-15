#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/params.h"
#include "selfdrive/pandad/panda.h"

void pandad_main_thread(std::string serial);

// deprecated devices
static const std::vector<cereal::PandaState::PandaType> SUPPORTED_PANDA_TYPES = {
  cereal::PandaState::PandaType::RED_PANDA,
  cereal::PandaState::PandaType::TRES,
  cereal::PandaState::PandaType::CUATRO,
};


class PandaSafety {
public:
  PandaSafety(Panda *panda) : panda_(panda) {}
  void configureSafetyMode(bool is_onroad);
  bool getOffroadMode();

  // Send diagnostic CAN frames while offroad, queued via the OffroadCanQueue param (used by the
  // auto door lock). Frames are drained one at a time with a gap (the body ECU drops a burst).
  // No-op onroad, where the real safety mode is active.
  void maybeSendOffroadCan(bool is_onroad);

  // Play the CAN script queued in the OffroadCanScript param (used by the hazard flash test).
  // Unlike the queue above, each frame carries its own pre-send delay, which is what ISO-TP
  // multi-frame requests and blink timing need. Call at the main loop rate for the resolution.
  // No-op onroad, where the real safety mode is active.
  void maybeSendOffroadCanScript(bool is_onroad);

private:
  void sendFrameViaElm327(uint16_t addr, uint8_t bus, const uint8_t *data, uint8_t dlc);
  void dropElm327(uint64_t now);

  void updateMultiplexingMode();
  std::vector<std::string> fetchCarParams();
  void setSafetyMode(const std::vector<std::string> &params_string);

  bool initialized_ = false;
  bool log_once_ = false;
  bool safety_configured_ = false;
  bool prev_obd_multiplexing_ = false;
  std::vector<std::string> offroad_records_;   // pending 12-byte CAN records to send, one per gap
  uint64_t last_offroad_send_ns_ = 0;
  std::vector<std::string> script_records_;   // pending 14-byte script records, each with its own delay
  uint64_t last_script_send_ns_ = 0;
  uint64_t last_script_poll_ns_ = 0;
  bool elm327_asserted_ = false;
  Panda *panda_;
  Params params_;
};
