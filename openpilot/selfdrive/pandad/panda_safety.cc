#include <algorithm>
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

void PandaSafety::maybeSendOffroadCan(bool is_onroad) {
  // Only ever touch the safety model offroad. Onroad the car-specific safety mode is active and
  // must not be disturbed, so the queue is intentionally ignored there.
  if (is_onroad) {
    offroad_records_.clear();
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

  // Hold off while a script is playing: its frames are timed, and interleaving diagnostic frames
  // from another feature could break an ISO-TP request in the middle.
  if (!script_records_.empty()) {
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

  // 0x750 is a UDS diagnostic address; ELM327 (no OBD multiplexing) is the least-privilege mode
  // that allows transmitting it. The offroad health loop re-asserts NO_OUTPUT afterwards.
  panda_->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1U);

  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto sendcan = evt.initSendcan(1);
  sendcan[0].setAddress(addr);
  sendcan[0].setDat(kj::arrayPtr((const uint8_t *)rec.data() + 4, dlc));
  sendcan[0].setSrc(bus);
  panda_->can_send(sendcan.asReader());

  // Revert immediately; don't leave the panda in an output-capable mode.
  panda_->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);

  LOGW("OffroadCan: sent frame 0x%x on bus %d via ELM327 (%zu queued)", addr, bus, offroad_records_.size());
}

// OffroadCanScript records: [delay_ms_hi, delay_ms_lo, addr_hi, addr_lo, bus, dlc, data[8]].
// delay_ms is how long to wait after the previous frame before sending this one; the queue above
// can only do a fixed gap, which is too coarse for an ISO-TP consecutive frame (tens of ms) and
// too fast for a blink (hundreds).
static constexpr size_t SCRIPT_RECORD_LEN = 14;

// The param only changes when someone presses a button, so don't hit the filesystem every frame.
static constexpr uint64_t SCRIPT_POLL_NS = 100000000ULL;  // 100 ms

// How long to stay in ELM327 after a frame before dropping back to NO_OUTPUT. Changing the safety
// model re-inits the panda's CAN cores, so reverting in the same breath as the send can flush the
// frame back out of the TX FIFO before it reaches the wire.
static constexpr uint64_t ELM327_LINGER_NS = 20000000ULL;  // 20 ms

void PandaSafety::sendFrameViaElm327(uint16_t addr, uint8_t bus, const uint8_t *data, uint8_t dlc) {
  // ELM327 (no OBD multiplexing) is the least-privilege mode that can transmit a diagnostic
  // address: it allows 8-byte frames on 0x600-0x7FF and nothing else.
  panda_->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1U);

  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto sendcan = evt.initSendcan(1);
  sendcan[0].setAddress(addr);
  sendcan[0].setDat(kj::arrayPtr(data, dlc));
  sendcan[0].setSrc(bus);
  panda_->can_send(sendcan.asReader());
}

void PandaSafety::maybeSendOffroadCanScript(bool is_onroad) {
  // Onroad the car-specific safety mode is active and must not be disturbed. Anything still
  // pending is dropped rather than resumed: a script is a one-shot test, not state to restore.
  if (is_onroad) {
    script_records_.clear();
    // configureSafetyMode() owns the safety model onroad, so just forget we ever set one.
    elm327_asserted_ = false;
    return;
  }

  uint64_t now = nanos_since_boot();
  if (now - last_script_poll_ns_ >= SCRIPT_POLL_NS) {
    last_script_poll_ns_ = now;
    std::string script = params_.get("OffroadCanScript");
    if (!script.empty()) {
      params_.remove("OffroadCanScript");
      // A new script replaces what is still pending instead of interleaving frames with it, so
      // pressing the button twice restarts the sequence.
      script_records_.clear();
      for (size_t i = 0; i + SCRIPT_RECORD_LEN <= script.size(); i += SCRIPT_RECORD_LEN) {
        script_records_.push_back(script.substr(i, SCRIPT_RECORD_LEN));
      }
      last_script_send_ns_ = now;
      LOGW("OffroadCanScript: starting %zu frames", script_records_.size());
    }
  }

  if (script_records_.empty()) {
    // Nothing left to play: hand the panda back once the last frame has had time to go out.
    dropElm327(now);
    return;
  }

  const std::string &rec = script_records_.front();
  uint64_t delay_ns = (((uint8_t)rec[0] << 8) | (uint8_t)rec[1]) * 1000000ULL;
  if (now - last_script_send_ns_ < delay_ns) {
    // Waiting out this frame's delay: nothing to send, so give the panda back in the meantime.
    dropElm327(now);
    return;
  }
  last_script_send_ns_ = now;
  elm327_asserted_ = true;

  uint16_t addr = ((uint8_t)rec[2] << 8) | (uint8_t)rec[3];
  uint8_t bus = (uint8_t)rec[4];
  uint8_t dlc = std::min((uint8_t)rec[5], (uint8_t)8);
  sendFrameViaElm327(addr, bus, (const uint8_t *)rec.data() + 6, dlc);
  script_records_.erase(script_records_.begin());

  if (script_records_.empty()) {
    LOGW("OffroadCanScript: done");
  }
}

void PandaSafety::dropElm327(uint64_t now) {
  // Don't leave the panda in an output-capable mode: back to NO_OUTPUT between frames too, not
  // just at the end of a script. The gaps in a script are longer than the linger, so this runs
  // after every frame. (The offroad health loop would get there within 100 ms regardless.)
  if (elm327_asserted_ && (now - last_script_send_ns_ >= ELM327_LINGER_NS)) {
    panda_->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
    elm327_asserted_ = false;
  }
}
