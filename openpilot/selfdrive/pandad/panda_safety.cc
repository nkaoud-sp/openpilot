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

  // Send policy is chosen from the frame at the head of the queue. Diagnostic frames to the
  // combination meter (0x7C0 - the turn-signal active test) are drained FAST: the whole command is
  // sent back-to-back in a single pass with one ELM327 flip, so pulse edges are tight and
  // consistent. Everything else (e.g. body ECU 0x750, used by auto-lock) keeps the 200 ms spacing
  // and one-frame-per-gap cadence, because that ECU drops back-to-back diagnostic frames.
  uint16_t head_addr = ((uint8_t)offroad_records_.front()[0] << 8) | (uint8_t)offroad_records_.front()[1];
  const bool fast = (head_addr == 0x7C0U);

  uint64_t now = nanos_since_boot();
  if (!fast && (now - last_offroad_send_ns_ < OFFROAD_CAN_GAP_NS)) {
    return;
  }
  last_offroad_send_ns_ = now;

  // Hold ELM327 for the whole batch (one flip), send, then revert to NO_OUTPUT.
  panda_->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1U);

  int sent = 0;
  const int max_batch = fast ? 8 : 1;  // fast: whole command per pass; else one frame per gap
  while (!offroad_records_.empty() && sent < max_batch) {
    const std::string &front = offroad_records_.front();
    uint16_t addr = ((uint8_t)front[0] << 8) | (uint8_t)front[1];
    if (sent > 0 && (addr == 0x7C0U) != fast) {
      break;  // don't mix fast/slow addresses in one batch
    }
    uint8_t bus = (uint8_t)front[2];
    uint8_t dlc = std::min((uint8_t)front[3], (uint8_t)8);

    MessageBuilder msg;
    auto evt = msg.initEvent();
    auto sendcan = evt.initSendcan(1);
    sendcan[0].setAddress(addr);
    sendcan[0].setDat(kj::arrayPtr((const uint8_t *)front.data() + 4, dlc));
    sendcan[0].setSrc(bus);
    panda_->can_send(sendcan.asReader());

    offroad_records_.erase(offroad_records_.begin());
    sent++;
  }

  // Revert immediately; don't leave the panda in an output-capable mode.
  panda_->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);

  LOGW("OffroadCan: sent %d frame(s) to 0x%x via ELM327 (%zu queued)", sent, head_addr, offroad_records_.size());
}
