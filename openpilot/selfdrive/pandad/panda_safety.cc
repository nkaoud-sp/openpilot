#include <algorithm>
#include <initializer_list>
#include <string>
#include <vector>

#include "selfdrive/pandad/pandad.h"
#include "openpilot/cereal/messaging/messaging.h"
#include "common/swaglog.h"
#include "common/timing.h"

void PandaSafety::configureSafetyMode(bool is_onroad) {
  if (is_onroad && !safety_configured_) {
    updateMultiplexingMode();

    auto car_params = fetchCarParams();
    if (!car_params.empty()) {
      LOGW("got %lu bytes CarParams", car_params[0].size());
      LOGW("got %lu bytes CarParamsSP", car_params[1].size());
      setSafetyMode(car_params);
      safety_configured_ = true;
    }
  } else if (!is_onroad) {
    initialized_ = false;
    safety_configured_ = false;
    log_once_ = false;
  }
}

void PandaSafety::updateMultiplexingMode() {
  // Initialize to ELM327 without OBD multiplexing for initial fingerprinting
  if (!initialized_) {
    prev_obd_multiplexing_ = false;
    panda_->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1U);
    initialized_ = true;
  }

  // Switch between multiplexing modes based on the OBD multiplexing request
  bool obd_multiplexing_requested = params_.getBool("ObdMultiplexingEnabled");
  if (obd_multiplexing_requested != prev_obd_multiplexing_) {
    const uint16_t safety_param = obd_multiplexing_requested ? 0U : 1U;
    panda_->set_safety_model(cereal::CarParams::SafetyModel::ELM327, safety_param);
    prev_obd_multiplexing_ = obd_multiplexing_requested;
    params_.putBool("ObdMultiplexingChanged", true);
  }
}

// TODO-SP: Use structs instead of vector
std::vector<std::string> PandaSafety::fetchCarParams() {
  if (!params_.getBool("FirmwareQueryDone")) {
    return {};
  }

  if (!log_once_) {
    LOGW("Finished FW query, Waiting for params to set safety model");
    log_once_ = true;
  }

  if (!params_.getBool("ControlsReady")) {
    return {};
  }
  return {params_.get("CarParams"), params_.get("CarParamsSP")};
}

// TODO-SP: Use structs instead of vector
void PandaSafety::setSafetyMode(const std::vector<std::string> &params_string) {
  AlignedBuffer aligned_buf;
  AlignedBuffer aligned_buf_sp;

  capnp::FlatArrayMessageReader cmsg(aligned_buf.align(params_string[0].data(), params_string[0].size()));
  cereal::CarParams::Reader car_params = cmsg.getRoot<cereal::CarParams>();

  capnp::FlatArrayMessageReader cmsg_sp(aligned_buf_sp.align(params_string[1].data(), params_string[1].size()));
  cereal::CarParamsSP::Reader car_params_sp = cmsg_sp.getRoot<cereal::CarParamsSP>();

  auto safety_configs = car_params.getSafetyConfigs();
  uint16_t alternative_experience = car_params.getAlternativeExperience();
  uint16_t safety_param_sp = car_params_sp.getSafetyParam();

  cereal::CarParams::SafetyModel safety_model = safety_configs[0].getSafetyModel();
  uint16_t safety_param = safety_configs[0].getSafetyParam();

  LOGW("setting safety model: %d, param: %d, alternative experience: %d, param_sp: %d", (int)safety_model, safety_param, alternative_experience, safety_param_sp);
  panda_->set_alternative_experience(alternative_experience, safety_param_sp);
  panda_->set_safety_model(safety_model, safety_param);
}

bool PandaSafety::getOffroadMode() {
  auto offroad_mode = params_.getBool("OffroadMode");
  return offroad_mode;
}

// Gap between offroad CAN frames. The body ECU drops back-to-back diagnostic frames, so they are
// sent one at a time this far apart. Tune here if some commands still don't land.
static constexpr uint64_t OFFROAD_CAN_GAP_NS = 200000000ULL;  // 200 ms
static constexpr uint64_t TURN_SIGNAL_ISOTP_FRAME_GAP_NS = 20000000ULL;  // 20 ms
static constexpr uint16_t TURN_SIGNAL_ADDR = 0x7C0;
static constexpr uint8_t TURN_SIGNAL_BUS = 0;

static std::string paddedFrame(std::initializer_list<uint8_t> bytes) {
  std::string dat;
  dat.reserve(8);
  for (uint8_t b : bytes) {
    dat.push_back((char)b);
  }
  dat.resize(8, '\0');
  return dat;
}

void PandaSafety::sendOffroadFrame(uint16_t addr, uint8_t bus, const std::string &dat) {
  panda_->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1U);

  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto sendcan = evt.initSendcan(1);
  sendcan[0].setAddress(addr);
  sendcan[0].setDat(kj::arrayPtr((const uint8_t *)dat.data(), dat.size()));
  sendcan[0].setSrc(bus);
  panda_->can_send(sendcan.asReader());

  panda_->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
}

void PandaSafety::maybeSendOffroadCan(bool is_onroad) {
  // Only ever touch the safety model offroad. Onroad the car-specific safety mode is active and
  // must not be disturbed, so the queue is intentionally ignored there.
  if (is_onroad) {
    offroad_records_.clear();
    return;
  }
  if (!turn_signal_sequence_.empty()) {
    return;
  }

  // Append newly requested frames to the pending queue.
  // OffroadCanQueue: 12-byte records [addr_hi, addr_lo, bus, dlc, data[8]].
  std::string queue = params_.get("OffroadCanQueue");
  if (!queue.empty()) {
    params_.remove("OffroadCanQueue");
    for (size_t i = 0; i + 12 <= queue.size(); i += 12) {
      offroad_records_.push_back(queue.substr(i, 12));
    }
  }

  if (offroad_records_.empty()) {
    return;
  }

  // Space the frames out: send at most one per OFFROAD_CAN_GAP_NS.
  uint64_t now = nanos_since_boot();
  if (now - last_offroad_send_ns_ < OFFROAD_CAN_GAP_NS) {
    return;
  }
  last_offroad_send_ns_ = now;

  std::string rec = offroad_records_.front();
  offroad_records_.erase(offroad_records_.begin());

  uint16_t addr = ((uint8_t)rec[0] << 8) | (uint8_t)rec[1];
  uint8_t bus = (uint8_t)rec[2];
  uint8_t dlc = std::min((uint8_t)rec[3], (uint8_t)8);

  sendOffroadFrame(addr, bus, rec.substr(4, dlc));

  LOGW("OffroadCan: sent frame 0x%x on bus %d via ELM327 (%zu queued)", addr, bus, offroad_records_.size());
}

void PandaSafety::startTurnSignalSequence() {
  turn_signal_sequence_.clear();
  turn_signal_sequence_idx_ = 0;
  next_turn_signal_sequence_ns_ = nanos_since_boot();

  auto add_frame = [this](const std::string &dat, uint64_t delay_ns) {
    turn_signal_sequence_.push_back({TURN_SIGNAL_ADDR, TURN_SIGNAL_BUS, dat, delay_ns});
  };
  auto add_command = [&](uint8_t signal_bits, bool on, double duration_s) {
    add_frame(paddedFrame({0x10, 0x0C, 0x2F, 0x29, 0x11, 0x03, 0x00, 0x00}), TURN_SIGNAL_ISOTP_FRAME_GAP_NS);
    uint8_t active_bits = on ? signal_bits : 0x00;
    uint64_t command_delay_ns = duration_s > 0.0 ? (uint64_t)std::max(0.0, (duration_s * 1e9) - TURN_SIGNAL_ISOTP_FRAME_GAP_NS) : 0;
    add_frame(paddedFrame({0x21, 0x00, active_bits, 0x00, 0x00, 0x00, signal_bits, 0x00}), command_delay_ns);
  };
  auto add_pattern = [&](uint8_t signal_bits, double on_s, double off_s, int repeats) {
    for (int i = 0; i < repeats; ++i) {
      add_command(signal_bits, true, on_s);
      add_command(signal_bits, false, off_s);
    }
  };

  add_frame(paddedFrame({0x02, 0x10, 0x03}), TURN_SIGNAL_ISOTP_FRAME_GAP_NS);
  add_pattern(0x10, 0.8, 0.4, 5);
  add_pattern(0x08, 0.4, 0.8, 5);
  add_pattern(0x18, 0.6, 0.6, 5);
  add_frame(paddedFrame({0x04, 0x2F, 0x29, 0x11, 0x00}), 0);

  LOGW("OffroadTurnSignalSequence: started (%zu frames)", turn_signal_sequence_.size());
}

void PandaSafety::maybeRunTurnSignalSequence(bool is_onroad) {
  if (is_onroad) {
    turn_signal_sequence_.clear();
    turn_signal_sequence_idx_ = 0;
    params_.remove("OffroadTurnSignalSequence");
    return;
  }

  if (params_.getBool("OffroadTurnSignalSequence")) {
    params_.remove("OffroadTurnSignalSequence");
    startTurnSignalSequence();
  }

  if (turn_signal_sequence_idx_ >= turn_signal_sequence_.size()) {
    return;
  }

  uint64_t now = nanos_since_boot();
  if (now < next_turn_signal_sequence_ns_) {
    return;
  }

  const TimedCanFrame &frame = turn_signal_sequence_[turn_signal_sequence_idx_++];
  sendOffroadFrame(frame.addr, frame.bus, frame.dat);
  next_turn_signal_sequence_ns_ = now + frame.delay_ns;

  if (turn_signal_sequence_idx_ >= turn_signal_sequence_.size()) {
    turn_signal_sequence_.clear();
    turn_signal_sequence_idx_ = 0;
    LOGW("OffroadTurnSignalSequence: finished");
  }
}
