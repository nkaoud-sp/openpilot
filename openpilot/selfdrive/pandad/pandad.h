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
  void maybeRunTurnSignalSequence(bool is_onroad);

private:
  struct TimedCanFrame {
    uint16_t addr;
    uint8_t bus;
    std::string dat;
    uint64_t delay_ns;
  };

  void updateMultiplexingMode();
  std::vector<std::string> fetchCarParams();
  void setSafetyMode(const std::vector<std::string> &params_string);
  void sendOffroadFrame(uint16_t addr, uint8_t bus, const std::string &dat);
  void startTurnSignalSequence();

  bool initialized_ = false;
  bool log_once_ = false;
  bool safety_configured_ = false;
  bool prev_obd_multiplexing_ = false;
  std::vector<std::string> offroad_records_;   // pending 12-byte CAN records to send, one per gap
  uint64_t last_offroad_send_ns_ = 0;
  std::vector<TimedCanFrame> turn_signal_sequence_;
  size_t turn_signal_sequence_idx_ = 0;
  uint64_t next_turn_signal_sequence_ns_ = 0;
  Panda *panda_;
  Params params_;
};
