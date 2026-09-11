#include <algorithm>

#include "selfdrive/pandad/pandad.h"
#include "cereal/messaging/messaging.h"
#include "common/swaglog.h"
#include "common/timing.h"

void PandaSafety::configureSafetyMode(bool is_onroad) {
  if (is_onroad && !safety_configured_) {
    updateMultiplexingMode();

    auto car_params = fetchCarParams();
    if (!car_params.empty()) {
      LOGW("got %lu bytes CarParams", car_params.size());
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
    for (int i = 0; i < pandas_.size(); ++i) {
      pandas_[i]->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1U);
    }
    initialized_ = true;
  }

  // Switch between multiplexing modes based on the OBD multiplexing request
  bool obd_multiplexing_requested = params_.getBool("ObdMultiplexingEnabled");
  if (obd_multiplexing_requested != prev_obd_multiplexing_) {
    for (int i = 0; i < pandas_.size(); ++i) {
      const uint16_t safety_param = (i > 0 || !obd_multiplexing_requested) ? 1U : 0U;
      pandas_[i]->set_safety_model(cereal::CarParams::SafetyModel::ELM327, safety_param);
    }
    prev_obd_multiplexing_ = obd_multiplexing_requested;
    params_.putBool("ObdMultiplexingChanged", true);
  }
}

std::string PandaSafety::fetchCarParams() {
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
  return params_.get("CarParams");
}

void PandaSafety::setSafetyMode(const std::string &params_string) {
  AlignedBuffer aligned_buf;
  capnp::FlatArrayMessageReader cmsg(aligned_buf.align(params_string.data(), params_string.size()));
  cereal::CarParams::Reader car_params = cmsg.getRoot<cereal::CarParams>();

  auto safety_configs = car_params.getSafetyConfigs();
  uint16_t alternative_experience = car_params.getAlternativeExperience();

  std::string starpilot_params_string = params_.get("StarPilotCarParams");

  AlignedBuffer starpilot_aligned_buf;
  capnp::FlatArrayMessageReader starpilot_cmsg(starpilot_aligned_buf.align(starpilot_params_string.data(), starpilot_params_string.size()));
  cereal::StarPilotCarParams::Reader starpilot_car_params = starpilot_cmsg.getRoot<cereal::StarPilotCarParams>();

  auto starpilot_safety_configs = starpilot_car_params.getSafetyConfigs();
  alternative_experience |= starpilot_car_params.getAlternativeExperience();
  for (int i = 0; i < pandas_.size(); ++i) {
    // Default to SILENT safety model if not specified
    cereal::CarParams::SafetyModel safety_model = cereal::CarParams::SafetyModel::SILENT;
    uint16_t safety_param = 0U;
    if (i < safety_configs.size()) {
      safety_model = safety_configs[i].getSafetyModel();
      safety_param = safety_configs[i].getSafetyParam();
    }

    if (i < starpilot_safety_configs.size()) {
      safety_param |= starpilot_safety_configs[i].getSafetyParam();
    }

    LOGW("Panda %d: setting safety model: %d, param: %d, alternative experience: %d", i, (int)safety_model, safety_param, alternative_experience);
    pandas_[i]->set_alternative_experience(alternative_experience);
    pandas_[i]->set_safety_model(safety_model, safety_param);
  }
}

static constexpr uint64_t OFFROAD_CAN_GAP_NS = 200000000ULL;

void PandaSafety::maybeSendOffroadCan(bool is_onroad) {
  if (is_onroad) {
    offroad_records_.clear();
    return;
  }

  std::string queue = params_.get("OffroadCanQueue");
  if (!queue.empty()) {
    params_.remove("OffroadCanQueue");
    for (size_t i = 0; i + 12 <= queue.size(); i += 12) {
      offroad_records_.push_back(queue.substr(i, 12));
    }
  }

  if (offroad_records_.empty() || pandas_.empty()) {
    return;
  }

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
  Panda *internal_panda = pandas_[0];

  internal_panda->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1U);

  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto sendcan = evt.initSendcan(1);
  sendcan[0].setAddress(addr);
  sendcan[0].setDat(kj::arrayPtr((const uint8_t *)rec.data() + 4, dlc));
  sendcan[0].setSrc(bus);
  internal_panda->can_send(sendcan.asReader());

  internal_panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);

  LOGW("OffroadCan: sent frame 0x%x on bus %d via ELM327 (%zu queued)", addr, bus, offroad_records_.size());
}
