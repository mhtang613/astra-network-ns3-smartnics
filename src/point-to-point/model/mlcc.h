// mlcc.h — standalone C++ port of the shipped NIC DCQCN rate models
// (v9_bf3 HistGBT, v9_pensando_gru_enhanced heteroscedastic GRU), loaded
// from the plain-text .mlcc files written by
// model/core/export/export_mlcc.py.
//
// Header-only so it can later be dropped into the ns-3 backend unchanged.
// The intended use there: one mlcc::Runner per sender node; every CNP
// arrival (CNP-flagged ACK) calls addCnp(t_us, 1); a 100 us timer calls
// step() and applies the returned rate to the node's QPs.
//
// Faithfulness notes (all mirror the Python training pipeline exactly):
//  - CNP arrivals are binned to 1 us; an arrival in bin b has timestamp
//    b + 0.5 us (training data is 1 us-binned counts). Arrivals MUST be
//    fed in nondecreasing time order.
//  - Features are computed in double, then cast to float32, matching the
//    float32 feature tables the models were trained/evaluated on. The GBT
//    then widens back to double (sklearn predict converts float32 input
//    to float64 before threshold comparison).
//  - t_norm is step/1500 within the trained 150 ms window and clamps at
//    1499/1500 beyond it (deployment choice; the models never saw more).
//  - The pensando equilibrium prior is the product of part-1 ratio curves
//    over registers moved from their defaults, times the QP-matched
//    control mean, /200 — computed from the RAW register values.
#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlcc {

struct GbtNode {
  int leaf, feat;
  double thr;
  int left, right;
  double val;
};

struct Model {
  std::string type, vendor;
  double step_us = 100.0;
  int nstep = 1500;
  double clip_max = 200.0;
  int n_temporal = 9, n_static = 0;
  std::vector<std::string> temporal, statics;
  std::vector<std::pair<std::string, double>> defaults;  // file order
  // gbt
  double baseline = 0.0;
  std::vector<std::vector<GbtNode>> trees;
  // gru_hetero
  double rate_scale = 100.0, c_1ms = 0.0;
  std::map<int, double> ctl_mean;
  std::map<std::string, std::vector<std::pair<double, double>>> curves;
  int hidden = 0, layers = 0;
  std::vector<float> norm_mu, norm_sd;
  std::map<std::string, std::vector<float>> tensors;  // row-major

  int nFeatures() const { return n_temporal + n_static; }

  static Model load(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open model file: " + path);
    Model m;
    std::string k;
    auto want = [&](bool ok, const char *what) {
      if (!ok) throw std::runtime_error(std::string("bad model file at ") +
                                        what);
    };
    while (f >> k) {
      if (k == "MLCC") { int v; f >> v; want(v == 1, "version"); }
      else if (k == "model") f >> m.type;
      else if (k == "vendor") f >> m.vendor;
      else if (k == "step_us") f >> m.step_us;
      else if (k == "nstep") f >> m.nstep;
      else if (k == "clip_max") f >> m.clip_max;
      else if (k == "n_temporal") f >> m.n_temporal;
      else if (k == "temporal") {
        m.temporal.resize(m.n_temporal);
        for (auto &s : m.temporal) f >> s;
      } else if (k == "n_static") f >> m.n_static;
      else if (k == "static") {
        m.statics.resize(m.n_static);
        for (auto &s : m.statics) f >> s;
      } else if (k == "n_defaults") { int n; f >> n; m.defaults.reserve(n); }
      else if (k == "default") {
        std::string name; double v; f >> name >> v;
        m.defaults.emplace_back(name, v);
      } else if (k == "baseline") f >> m.baseline;
      else if (k == "n_trees") { int n; f >> n; m.trees.reserve(n); }
      else if (k == "tree") {
        int idx, n; f >> idx >> n;
        std::vector<GbtNode> t(n);
        for (auto &nd : t)
          f >> nd.leaf >> nd.feat >> nd.thr >> nd.left >> nd.right >> nd.val;
        m.trees.push_back(std::move(t));
      } else if (k == "rate_scale") f >> m.rate_scale;
      else if (k == "C_1ms") f >> m.c_1ms;
      else if (k == "ctl_mean") {
        int n; f >> n;
        for (int i = 0; i < n; i++) { int q; double v; f >> q >> v;
                                      m.ctl_mean[q] = v; }
      } else if (k == "n_curves") { int n; f >> n; (void)n; }
      else if (k == "curve") {
        std::string reg; int n; f >> reg >> n;
        auto &c = m.curves[reg];
        c.resize(n);
        for (auto &p : c) f >> p.first >> p.second;
      } else if (k == "hidden") f >> m.hidden;
      else if (k == "layers") f >> m.layers;
      else if (k == "norm_mu" || k == "norm_sd") {
        int n; f >> n;
        auto &v = (k == "norm_mu") ? m.norm_mu : m.norm_sd;
        v.resize(n);
        for (auto &x : v) f >> x;
      } else if (k == "tensor") {
        std::string name; int r, c; f >> name >> r >> c;
        auto &v = m.tensors[name];
        v.resize((size_t)r * c);
        for (auto &x : v) f >> x;
      } else if (k == "end") break;
      else throw std::runtime_error("unknown model-file key: " + k);
    }
    want(m.type == "gbt" || m.type == "gru_hetero", "model type");
    want((int)m.statics.size() == m.n_static, "static names");
    return m;
  }
};

// Streaming computation of the 9 temporal features at 100 us steps from
// 1 us-binned CNP arrivals. Matches train_v8_blackbox.run_features().
struct FeatureStream {
  int nstep_window;          // t_norm denominator (1500 = 150 ms)
  int cur_step = 0;
  long long cum = 0;
  double last_ts = -1.0, prev_arrival = -1e18;
  int bin_cnt[100];          // per-us counts within the current step
  int merged = 0;            // arrivals with gap < 4 us, current step
  int ring10[10], ring100[100];
  long long sum10 = 0, sum100 = 0;
  int pos10 = 0, pos100 = 0;

  explicit FeatureStream(int nstep) : nstep_window(nstep) {
    for (auto &v : bin_cnt) v = 0;
    for (auto &v : ring10) v = 0;
    for (auto &v : ring100) v = 0;
  }

  // CNP arrival(s) at absolute time t_us; must be nondecreasing in time.
  // An arrival at/after the current step's end (e.g. an ns-3 event tied
  // with the step timer at the exact boundary) is buffered and fed into
  // the step it belongs to.
  void addCnp(double t_us, int count = 1) {
    int bin = (int)std::floor(t_us);
    if (bin >= (cur_step + 1) * 100) {
      pending.emplace_back(bin, count);
      return;
    }
    feed(bin, count);
  }

 private:
  std::vector<std::pair<int, int>> pending;

  void feed(int bin, int count) {
    int idx = bin - cur_step * 100;
    if (idx < 0 || idx >= 100)
      throw std::runtime_error("addCnp outside current step");
    double ts = bin + 0.5;
    bin_cnt[idx] += count;
    if (ts - prev_arrival < 4.0) merged += 1;  // gap to previous arrival
    merged += count - 1;                       // same-bin arrivals, gap 0
    prev_arrival = ts;
    last_ts = ts;
  }

  // Close the current 100 us step and emit its 9 features (double; caller
  // casts to float32 to match training).
 public:
  void step(double *f) {
    int cnt = 0, busy = 0, mx = 0;
    for (int i = 0; i < 100; i++) {
      cnt += bin_cnt[i];
      if (bin_cnt[i] > 0) {
        busy++;
        if (bin_cnt[i] > mx) mx = bin_cnt[i];
      }
      bin_cnt[i] = 0;
    }
    cum += cnt;
    sum10 += cnt - ring10[pos10];
    ring10[pos10] = cnt;
    pos10 = (pos10 + 1) % 10;
    sum100 += cnt - ring100[pos100];
    ring100[pos100] = cnt;
    pos100 = (pos100 + 1) % 100;

    double step_end = (cur_step + 1) * 100.0;
    double since = 50.0;
    if (last_ts >= 0.0)
      since = std::min((step_end - last_ts) / 1000.0, 50.0);
    int tstep = std::min(cur_step, nstep_window - 1);

    f[0] = std::log1p((double)cnt);
    f[1] = (double)busy;
    f[2] = (double)mx;
    f[3] = (double)merged;
    f[4] = std::log1p((double)sum10);
    f[5] = std::log1p((double)sum100);
    f[6] = std::log1p((double)cum);
    f[7] = since;
    f[8] = (double)tstep / nstep_window;
    merged = 0;
    cur_step++;
    if (!pending.empty()) {
      std::vector<std::pair<int, int>> left;
      for (const auto &pc : pending) {
        if (pc.first < (cur_step + 1) * 100) feed(pc.first, pc.second);
        else left.push_back(pc);
      }
      pending.swap(left);
    }
  }
};

// One model instance driving one sender node.
class Runner {
 public:
  Runner(const Model &m, const std::map<std::string, double> &reg_overrides,
         int qps, double cap_gbps)
      : m_(m), fs_(m.nstep) {
    regs_.clear();
    for (const auto &kv : m_.defaults) regs_[kv.first] = kv.second;
    for (const auto &kv : reg_overrides) {
      if (!regs_.count(kv.first))
        throw std::runtime_error("unknown register: " + kv.first);
      regs_[kv.first] = kv.second;
    }
    qps_ = qps;
    cap_ = cap_gbps;
    buildStatic();
    if (m_.type == "gru_hetero")
      h_.assign((size_t)m_.layers * m_.hidden, 0.0f);
  }

  // Change the active QP count mid-run (rebuilds the static features,
  // including the prior; keeps CNP history and GRU state).
  void setQps(int qps) {
    if (qps == qps_) return;
    qps_ = qps;
    buildStatic();
  }

  void addCnp(double t_us, int count = 1) { fs_.addCnp(t_us, count); }

  // Close the current 100 us step; returns the rate (Gb/s, clipped).
  // For gru_hetero, *sigma_out (if given) gets the per-step sigma (Gb/s).
  double step(double *sigma_out = nullptr) {
    double tf[9];
    fs_.step(tf);
    if (m_.type == "gbt") {
      std::vector<double> x(m_.nFeatures());
      for (int i = 0; i < m_.n_temporal; i++) x[i] = (double)(float)tf[i];
      for (int j = 0; j < m_.n_static; j++)
        x[m_.n_temporal + j] = (double)static_f_[j];
      double p = m_.baseline;
      for (const auto &t : m_.trees) {
        int i = 0;
        while (!t[i].leaf) i = x[t[i].feat] <= t[i].thr ? t[i].left
                                                        : t[i].right;
        p += t[i].val;
      }
      if (sigma_out) *sigma_out = 0.0;
      return clip(p);
    }
    // gru_hetero
    std::vector<float> x(m_.nFeatures());
    for (int i = 0; i < m_.n_temporal; i++) x[i] = (float)tf[i];
    for (int j = 0; j < m_.n_static; j++) x[m_.n_temporal + j] = static_f_[j];
    for (int i = 0; i < m_.nFeatures(); i++)
      x[i] = (x[i] - m_.norm_mu[i]) / m_.norm_sd[i];
    const float *inp = x.data();
    int in_dim = m_.nFeatures();
    for (int L = 0; L < m_.layers; L++) {
      gruLayer(L, inp, in_dim, &h_[(size_t)L * m_.hidden]);
      inp = &h_[(size_t)L * m_.hidden];
      in_dim = m_.hidden;
    }
    const auto &hw = m_.tensors.at("head.weight");
    const auto &hb = m_.tensors.at("head.bias");
    float mu = hb[0], lv = hb[1];
    for (int j = 0; j < m_.hidden; j++) {
      mu += hw[j] * inp[j];
      lv += hw[m_.hidden + j] * inp[j];
    }
    if (lv < -8.0f) lv = -8.0f;
    if (lv > 2.0f) lv = 2.0f;
    if (sigma_out)
      *sigma_out = std::sqrt(std::exp((double)lv)) * m_.rate_scale;
    return clip((double)mu * m_.rate_scale);
  }

  double prior() const { return prior_; }
  const Model &model() const { return m_; }

 private:
  double clip(double r) const {
    return r < 0.0 ? 0.0 : (r > m_.clip_max ? m_.clip_max : r);
  }

  void buildStatic() {
    static_f_.clear();
    for (const auto &name : m_.statics) {
      double v;
      if (name == "qps") v = (double)qps_;
      else if (name == "cap_norm") v = cap_ / m_.clip_max;
      else if (name == "eq_prior") v = prior_ = equilibriumPrior();
      else v = std::log10(1.0 + regs_.at(name));
      static_f_.push_back((float)v);
    }
  }

  // product of part-1 ratio curves over moved registers x ctl mean / 200
  double equilibriumPrior() const {
    double r = 1.0;
    for (const auto &kv : m_.defaults) {
      double v = regs_.at(kv.first);
      if (std::fabs(v - kv.second) > 0.5 * std::max(1.0, 0.01 * v)) {
        auto it = m_.curves.find(kv.first);
        if (it == m_.curves.end()) continue;
        const auto &c = it->second;
        double lv = std::log10(1.0 + v);
        double lo = std::log10(1.0 + c.front().first);
        double hi = std::log10(1.0 + c.back().first);
        lv = std::min(std::max(lv, lo), hi);
        double ratio = c.back().second;
        for (size_t i = 0; i + 1 < c.size(); i++) {
          double x0 = std::log10(1.0 + c[i].first);
          double x1 = std::log10(1.0 + c[i + 1].first);
          if (lv <= x1) {
            ratio = x1 == x0 ? c[i].second
                             : c[i].second + (c[i + 1].second - c[i].second) *
                                                 (lv - x0) / (x1 - x0);
            break;
          }
        }
        r *= ratio;
      }
    }
    auto it = m_.ctl_mean.find(qps_);
    if (it == m_.ctl_mean.end())  // nearest trained QP count
      it = m_.ctl_mean.lower_bound(qps_) == m_.ctl_mean.end()
               ? std::prev(m_.ctl_mean.end())
               : m_.ctl_mean.lower_bound(qps_);
    return r * it->second / m_.clip_max;
  }

  void gruLayer(int L, const float *x, int in_dim, float *h) {
    const int H = m_.hidden;
    const auto &wih = m_.tensors.at("gru.weight_ih_l" + std::to_string(L));
    const auto &whh = m_.tensors.at("gru.weight_hh_l" + std::to_string(L));
    const auto &bih = m_.tensors.at("gru.bias_ih_l" + std::to_string(L));
    const auto &bhh = m_.tensors.at("gru.bias_hh_l" + std::to_string(L));
    std::vector<float> gi(3 * H), gh(3 * H);
    for (int g = 0; g < 3 * H; g++) {
      float a = bih[g];
      const float *w = &wih[(size_t)g * in_dim];
      for (int j = 0; j < in_dim; j++) a += w[j] * x[j];
      gi[g] = a;
      float b = bhh[g];
      const float *u = &whh[(size_t)g * H];
      for (int j = 0; j < H; j++) b += u[j] * h[j];
      gh[g] = b;
    }
    for (int j = 0; j < H; j++) {
      float r = sigm(gi[j] + gh[j]);
      float z = sigm(gi[H + j] + gh[H + j]);
      float n = std::tanh(gi[2 * H + j] + r * gh[2 * H + j]);
      h[j] = (1.0f - z) * n + z * h[j];
    }
  }

  static float sigm(float a) { return 1.0f / (1.0f + std::exp(-a)); }

  const Model &m_;
  FeatureStream fs_;
  std::map<std::string, double> regs_;
  std::vector<float> static_f_;
  std::vector<float> h_;
  int qps_ = 1;
  double cap_ = 200.0;
  double prior_ = -1.0;
};

}  // namespace mlcc
