// clus_migration.C
// ROOT macro: event-by-event migration of nclust between two clustering time
// windows
//
// Purpose
//   Compare the SAME physics events (by TTree entry index) in two replay
//   outputs that differ only in the clustering time window (ns). Build a
//   migration matrix:
//     M(X -> Y) = #events with nclust_loose = X and nclust_tight = Y
//
// Conventions (matching clus_basic_hists.C)
//   - Rootfiles live in: rootfiles/
//   - Output PNGs go to: img/img_jan2026/
//   - CSV summary goes to: csv/clus_migration_stats.csv
//
// IMPORTANT NAMING NOTE
//   Your ns clustering window is encoded in the ROOT filename, but it is
//   (mis)labeled as "sec":
//     rootfiles/nps_hms_coin_<NS>sec_<RUN>_<SEG>_<TAG1>_<TAG2>.root
//   So nsLoose/nsTight are "ns windows" even though the filename says "sec".
//
// Run examples
//   root -l 'clus_migration.C(3013,5,0,5,1)'
//   root -l 'clus_migration.C(3013,5,0,4,2,200000)'
//
// Arguments
//   run        : run number
//   seg        : segment index
//   nsLoose    : "loose" clustering time window in ns (encoded in filename as
//   "<NS>sec") nsTight    : "tight" clustering time window in ns (encoded in
//   filename as "<NS>sec") maxEntries : cap on entries compared (-1 for all)
//   nclustMax  : clamp nclust to [0, nclustMax] for the matrix
//   batch      : batch mode for ROOT

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1I.h"
#include "TH2I.h"
#include "TLatex.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"
#include "TPad.h"
#include "TAxis.h"

static const int ROWS = 36;
static const int COLS = 30;
static const int N_BLOCKS = ROWS * COLS;
static const double EREF_MIN = 0.10;   // GeV: require loose energy above this for fractional plots

// -------------------------------
// Input path helper
// -------------------------------
// NOTE: The first numeric field is nsWindow, but filenames say "<NS>sec".
static std::string BuildInputPathNS(int run, int nsWindow, int seg,
                                    int tag1 = 1, int tag2 = -1) {
  return Form("rootfiles/nps_hms_coin_%dsec_%d_%d_%d_%d.root", nsWindow, run,
              seg, tag1, tag2);
}

// -------------------------------
// Plot stamp (wrapped + compact)
// -------------------------------
static void DrawWrappedLatex(TLatex &lat, double x, double &y, double dy,
                            int maxChars, const std::string &s) {
  if ((int)s.size() <= maxChars) {
    lat.DrawLatex(x, y, s.c_str());
    y -= dy;
    return;
  }

  // simple word-wrap
  size_t start = 0;
  while (start < s.size()) {
    size_t end = std::min(start + (size_t)maxChars, s.size());

    // try to break at last space before end
    if (end < s.size()) {
      size_t sp = s.rfind(' ', end);
      if (sp != std::string::npos && sp > start + 10) end = sp;
    }

    std::string line = s.substr(start, end - start);
    while (!line.empty() && line.front() == ' ')
      line.erase(line.begin());

    lat.DrawLatex(x, y, line.c_str());
    y -= dy;

    start = end;
    while (start < s.size() && s[start] == ' ')
      start++;
  }
}

static void StampMigration(int run, int seg, int nsLoose,
                           int nsTight, double edtmtdc_max, double hdelta_min,
                           double hdelta_max, double hcaltot_min,
                           double hcernpe_min, Long64_t maxEntries,
                           Long64_t nPass, Long64_t nSame) {

  TLatex lat;
  lat.SetNDC(true);
  lat.SetTextAlign(31);    // right-justified
  lat.SetTextSize(0.024);  // smaller

  const double x = 0.92;
  double y = 0.89;
  const double dy = 0.034; // slightly tighter than before
  const int WRAP = 70;

  lat.DrawLatex(
      x, y,
      Form("Run %d | seg %d | nclust migration: %d ns #rightarrow %d ns",
           run, seg, nsLoose, nsTight));
  y -= dy;

  std::string cuts = Form(
      "Cuts: EDTM<%.2f | dp[%.0f,%.0f] | etot>%.2f | npe>%.1f ",
      edtmtdc_max, hdelta_min, hdelta_max, hcaltot_min, hcernpe_min);
  DrawWrappedLatex(lat, x, y, dy, WRAP, cuts);

// Line 1: description only
DrawWrappedLatex(lat, x, y, dy, WRAP,
                 "Events passing cuts in BOTH trees:");

// Line 2: numbers
double fracSame = (nPass > 0) ? double(nSame) / double(nPass) : 0.0;

std::string statsNums = Form(
    "Total: %lld | unchanged: %lld | fraction: %.4f",
    (long long)nPass, (long long)nSame, fracSame);

DrawWrappedLatex(lat, x, y, dy, WRAP, statsNums);

}

// -------------------------------
// Save canvas
// -------------------------------
static void SaveCanvasPNG(TCanvas *c, int run, int seg, int nsLoose, int nsTight, const char *tag) {
  std::string out = Form("img/img_jan2026/%d_ns%d_to_ns%d_%s.png", run, nsLoose, nsTight, tag);
  c->SaveAs(out.c_str());
  std::cout << "Saved: " << out << "\n";
}


// -------------------------------
// Draw 1D delta hist with header-pad style + save
// -------------------------------
static void DrawAndSaveDelta1D(TH1 *h, int run, int seg, int nsLoose, int nsTight,
                               const char *titleText, const char *tag,
                               double edtm_max, double hdelta_min, double hdelta_max,
                               double hcaltot_min, double hcernpe_min,
                               Long64_t nPass, Long64_t nSame,
                               bool logy = true,
                               bool setXRange = false, double xmin = 0, double xmax = 0) {
  if (!h) return;

  TCanvas c(Form("c_%s", tag), "", 900, 700);

  TPad *pHead = new TPad(Form("pHead_%s", tag), Form("pHead_%s", tag), 0.0, 0.84, 1.0, 1.0);
  TPad *pMain = new TPad(Form("pMain_%s", tag), Form("pMain_%s", tag), 0.0, 0.0,  1.0, 0.84);

  pHead->SetFillStyle(0);
  pHead->SetBorderMode(0);
  pHead->SetBottomMargin(0.0);
  pHead->SetTopMargin(0.25);
  pHead->SetLeftMargin(0.12);
  pHead->SetRightMargin(0.06);

  pMain->SetFillStyle(0);
  pMain->SetBorderMode(0);
  pMain->SetTopMargin(0.02);
  pMain->SetBottomMargin(0.12);
  pMain->SetLeftMargin(0.12);
  pMain->SetRightMargin(0.06);

  pHead->Draw();
  pMain->Draw();

  // ---- main pad
  pMain->cd();
  gPad->SetTicks(1, 0);
  if (logy) gPad->SetLogy();

  h->SetTitle("");
  h->SetLineWidth(2);
  if (logy) h->SetMinimum(0.5);

  if (setXRange) h->GetXaxis()->SetRangeUser(xmin, xmax);

  // use integer ticks if this is an integer delta hist
  if (dynamic_cast<TH1I *>(h) != nullptr) {
    TAxis *ax = h->GetXaxis();
    ax->SetNdivisions(510);
    ax->SetLabelSize(0.035);
    ax->SetTitleSize(0.04);
    ax->SetTitleOffset(1.1);
  }

  h->Draw("HIST");
  gPad->Update();

  // ---- header pad
  pHead->cd();
  {
    TLatex lat;
    lat.SetNDC(true);

    // title
    lat.SetTextAlign(23);
    lat.SetTextSize(0.26);
    lat.DrawLatex(0.5, 0.95, titleText);

    // stamp
    lat.SetTextAlign(11);
    lat.SetTextSize(0.15);
    double x = 0.12, y = 0.55, dy = 0.20;

    lat.DrawLatex(x, y,
                  Form("Run %d | seg %d | nclust migration: %d ns #rightarrow %d ns",
                       run, seg, nsLoose, nsTight));
    y -= dy;
    lat.DrawLatex(x, y,
                  Form("Cuts: EDTM<%.2f | dp[%.0f,%.0f] | etot>%.2f | npe>%.1f",
                       edtm_max, hdelta_min, hdelta_max, hcaltot_min, hcernpe_min));
    y -= dy;

    double fracSame = (nPass > 0) ? double(nSame) / double(nPass) : 0.0;
    lat.DrawLatex(x, y,
                  Form("Events passing cuts in BOTH trees: Total %lld | unchanged %lld | frac %.4f",
                       (long long)nPass, (long long)nSame, fracSame));
  }

  c.cd();
  SaveCanvasPNG(&c, run, seg, nsLoose, nsTight, tag);
}

// -------------------------------
// CSV helpers
// -------------------------------
static void EnsureCSVHeader(const std::string &csvPath) {
  std::ifstream fin(csvPath);
  bool needHeader =
      (!fin.good()) || fin.peek() == std::ifstream::traits_type::eof();
  fin.close();

  if (needHeader) {
    std::ofstream fout(csvPath, std::ios::app);
    fout << "run,seg,nsLoose,nsTight,maxEntries,"
            "edtmtdc_max,hdelta_min,hdelta_max,hcaltot_min,hcernpe_min,"
            "nPass,nSame,fSame,"
            "fromN,toN,count,fraction_of_pass\n";
  }
}

static void AppendCSVRow(const std::string &csvPath, int run,
                         int seg, int nsLoose, int nsTight,
                         long long maxEntries, double edtmtdc_max,
                         double hdelta_min, double hdelta_max,
                         double hcaltot_min, double hcernpe_min,
                         long long nPass, long long nSame, int fromN, int toN,
                         long long count) {
  std::ofstream fout(csvPath, std::ios::app);
  double fSame = (nPass > 0) ? double(nSame) / double(nPass) : NAN;
  double frac = (nPass > 0) ? double(count) / double(nPass) : NAN;

  fout << run << "," << seg << "," << nsLoose << "," << nsTight
       << "," << maxEntries << "," << edtmtdc_max << "," << hdelta_min << ","
       << hdelta_max << "," << hcaltot_min << "," << hcernpe_min << "," << nPass
       << "," << nSame << "," << std::setprecision(8) << fSame << "," << fromN
       << "," << toN << "," << count << "," << std::setprecision(8) << frac
       << "\n";
}

struct EventClusSummary {
  int nclust = 0;
  int nblk_all = 0;        // timing-qualified blocks in ANY cluster
  int nblk_sel = 0;        // timing-qualified blocks in SELECTED clusters
  double Esum_sel = 0.0;   // sum clusE over selected clusters
  double Emax_sel = 0.0;   // max clusE among selected clusters
};

static EventClusSummary SummarizeEvent(int nclust,
                                       const double *clusE,
                                       const double *clusT,
                                       const double *block_clusterID,
                                       const double *goodAdcTdcDiffTime,
                                       double ecut, double tmin, double tmax,
                                       double dt_abs_max) {
  EventClusSummary s;
  s.nclust = nclust;

  if (nclust <= 0) return s;

  std::vector<char> sel(nclust, 0);
  for (int cid = 0; cid < nclust; cid++) {
    double e = clusE[cid];
    double t = clusT[cid];
    if (e >= ecut && t >= tmin && t <= tmax) {
      sel[cid] = 1;
      s.Esum_sel += e;
      if (e > s.Emax_sel) s.Emax_sel = e;
    }
  }

  for (int ib = 0; ib < N_BLOCKS; ib++) {
    int cid = (int)block_clusterID[ib];
    if (cid < 0 || cid >= nclust) continue;

    double dt = goodAdcTdcDiffTime[ib];
    if (!std::isfinite(dt)) continue;
    if (std::fabs(dt) > dt_abs_max) continue;

    s.nblk_all++;
    if (sel[cid]) s.nblk_sel++;
  }

  return s;
}

// ===============================
// MAIN MACRO
// ===============================
void clus_migration(int run = 3013, int seg = 0,
                    int nsLoose = 5, int nsTight = 1, Long64_t maxEntries = -1,
                    int nclustMax = 30, bool batch = true) {
  if (batch)
    gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(0);

  // ---- event-level cuts (same as clus_basic_hists.C)
  const double EDTM_TDC_MAX = 0.1;
  const double HDELTA_MIN = -12.0;
  const double HDELTA_MAX = 12.0;
  const double HCALTOT_MIN = 0.6;
  const double HCERNPE_MIN = 1.0;

  const double CLUS_ECUT = 0.60;
  const double CLUS_TMIN = 130.0;
  const double CLUS_TMAX = 170.0;
  const double DT_ABS_MAX = 1e6;  // same sentinel logic

  // ---- open files
  std::string pathLoose = BuildInputPathNS(run, nsLoose, seg);
  std::string pathTight = BuildInputPathNS(run, nsTight, seg);

  std::cout << "Opening (loose): " << pathLoose << "\n";
  std::cout << "Opening (tight): " << pathTight << "\n";

  TFile *fL = TFile::Open(pathLoose.c_str(), "READ");
  if (!fL || fL->IsZombie())
    return;

  TFile *fT = TFile::Open(pathTight.c_str(), "READ");
  if (!fT || fT->IsZombie())
    return;

  TTree *tL = (TTree *)fL->Get("T");
  TTree *tT = (TTree *)fT->Get("T");
  if (!tL || !tT)
    return;

// ===============================
// BRANCH SETUP (enable only used branches)
// ===============================

tL->SetBranchStatus("*", 0);
tT->SetBranchStatus("*", 0);

// --- event key (for matching)
tL->SetBranchStatus("fEvtHdr.fEvtNum", 1);
tT->SetBranchStatus("fEvtHdr.fEvtNum", 1);

// --- event-level cuts
tL->SetBranchStatus("T.hms.hEDTM_tdcTimeRaw", 1);
tL->SetBranchStatus("H.gtr.dp", 1);
tL->SetBranchStatus("H.cal.etotnorm", 1);
tL->SetBranchStatus("H.cer.npeSum", 1);

tT->SetBranchStatus("T.hms.hEDTM_tdcTimeRaw", 1);
tT->SetBranchStatus("H.gtr.dp", 1);
tT->SetBranchStatus("H.cal.etotnorm", 1);
tT->SetBranchStatus("H.cer.npeSum", 1);

// --- cluster-level
tL->SetBranchStatus("NPS.cal.nclust", 1);
tL->SetBranchStatus("NPS.cal.clusE", 1);
tL->SetBranchStatus("NPS.cal.clusT", 1);

tT->SetBranchStatus("NPS.cal.nclust", 1);
tT->SetBranchStatus("NPS.cal.clusE", 1);
tT->SetBranchStatus("NPS.cal.clusT", 1);

// --- block-level (for blocks-per-cluster and timing-qualified blocks)
tL->SetBranchStatus("NPS.cal.fly.block_clusterID", 1);
tL->SetBranchStatus("NPS.cal.fly.goodAdcTdcDiffTime", 1);

tT->SetBranchStatus("NPS.cal.fly.block_clusterID", 1);
tT->SetBranchStatus("NPS.cal.fly.goodAdcTdcDiffTime", 1);

// ---- set addresses
UInt_t evtNumL = 0, evtNumT = 0;

double edtmL = 0, dpL = 0, etotL = 0, npeL = 0, nclustL_d = 0;
double edtmT = 0, dpT = 0, etotT = 0, npeT = 0, nclustT_d = 0;

// arrays: match clus_basic_hists.C conventions
double clusE_L[10000], clusT_L[10000];
double clusE_T[10000], clusT_T[10000];

double block_clusterID_L[N_BLOCKS];
double goodAdcTdcDiffTime_L[N_BLOCKS];

double block_clusterID_T[N_BLOCKS];
double goodAdcTdcDiffTime_T[N_BLOCKS];

// --- key
tL->SetBranchAddress("fEvtHdr.fEvtNum", &evtNumL);
tT->SetBranchAddress("fEvtHdr.fEvtNum", &evtNumT);

// --- cuts
tL->SetBranchAddress("T.hms.hEDTM_tdcTimeRaw", &edtmL);
tL->SetBranchAddress("H.gtr.dp", &dpL);
tL->SetBranchAddress("H.cal.etotnorm", &etotL);
tL->SetBranchAddress("H.cer.npeSum", &npeL);

tT->SetBranchAddress("T.hms.hEDTM_tdcTimeRaw", &edtmT);
tT->SetBranchAddress("H.gtr.dp", &dpT);
tT->SetBranchAddress("H.cal.etotnorm", &etotT);
tT->SetBranchAddress("H.cer.npeSum", &npeT);

// --- cluster-level
tL->SetBranchAddress("NPS.cal.nclust", &nclustL_d);
tL->SetBranchAddress("NPS.cal.clusE", clusE_L); // <-- no &
tL->SetBranchAddress("NPS.cal.clusT", clusT_L); // <-- no &

tT->SetBranchAddress("NPS.cal.nclust", &nclustT_d);
tT->SetBranchAddress("NPS.cal.clusE", clusE_T); // <-- no &
tT->SetBranchAddress("NPS.cal.clusT", clusT_T); // <-- no &

// --- block-level
tL->SetBranchAddress("NPS.cal.fly.block_clusterID", block_clusterID_L);         // <-- no &
tL->SetBranchAddress("NPS.cal.fly.goodAdcTdcDiffTime", goodAdcTdcDiffTime_L);   // <-- no &

tT->SetBranchAddress("NPS.cal.fly.block_clusterID", block_clusterID_T);         // <-- no &
tT->SetBranchAddress("NPS.cal.fly.goodAdcTdcDiffTime", goodAdcTdcDiffTime_T);   // <-- no &

  // ===============================
  // HISTOGRAMS
  // ===============================
  TH2I *hMig =
      new TH2I("hMig",
               Form("nclust migration;N_{clust} @ %d ns;N_{clust} @ %d ns",
                    nsLoose, nsTight),
               nclustMax + 1, -0.5, nclustMax + 0.5, nclustMax + 1, -0.5,
               nclustMax + 0.5);

  // ΔNclust = N_tight − N_loose (integer). Use one bin per integer in a focused
  // range.
  const int DMIN = -50;
  const int DMAX = 50;
  const int NBINS_D = DMAX - DMIN + 1; // bin width = 1

  TH1I *hDelta =
      new TH1I("hDelta",
               Form("#Delta N_{clust} (tight - loose);#Delta N_{clust};Events"),
               NBINS_D, DMIN - 0.5, DMAX + 0.5);
  
  TH1I *hDeltaNblkAll = new TH1I("hDeltaNblkAll",
    "#Delta N_{blk} all (tight - loose);#Delta N_{blk};Events", 21, -10.5, 10.5);
  
  TH1I *hDeltaNblkSel = new TH1I("hDeltaNblkSel",
    "#Delta N_{blk} selected (tight - loose);#Delta N_{blk};Events", 41, -20.5, 20.5);
  
  TH1F *hDeltaEsumSel = new TH1F("hDeltaEsumSel",
    "#Delta E_{sum} selected (tight - loose);#Delta E_{sum} (GeV);Events", 200, -2.0, 2.0);
  
  TH1F *hDeltaEmaxSel = new TH1F("hDeltaEmaxSel",
    "#Delta E_{max} selected (tight - loose);#Delta E_{max} (GeV);Events", 200, -2.0, 2.0);

  TH1F *hFracEsumSel = new TH1F("hFracEsumSel",
    "Fractional #Delta E_{sum} selected (tight-loose)/loose;(#Delta E_{sum})/E_{sum,loose};Events",
    240, -1.2, 1.2);

  TH1F *hFracEmaxSel = new TH1F("hFracEmaxSel",
    "Fractional #Delta E_{max} selected (tight-loose)/loose;(#Delta E_{max})/E_{max,loose};Events",
    240, -1.2, 1.2);

  // ===============================
  // EVENT LOOP
  // building a lookup table to match the same physics event across two files
  // even when entry indices don’t line up
  // ===============================
  
  // evtNum -> per-event cluster summary (after cuts)
  std::unordered_map<UInt_t, EventClusSummary> looseEvt;
  
  // reserve to avoid rehashing during fill (performance)
  looseEvt.reserve((size_t)std::min<Long64_t>(
      tL->GetEntries(),
      2000000LL));
  
  Long64_t nLuse = tL->GetEntries();
  if (maxEntries > 0 && maxEntries < nLuse)
    nLuse = maxEntries;

  long long nLoosePass = 0; // counters for dx; how many evts pass cuts
  long long nLooseDup = 0; // counters for dx; how often the same evtNum appears
                           // twice (should be 0)
  long long nTightPass = 0, nMatchedPass = 0, nTightOnly = 0, nSame = 0;

  // ---- delta diagnostics (event-level truth)
  long long nNegDelta = 0;
  int minDelta = 999999;
  int maxDelta = -999999;
  int nPrintNeg = 0; // print first few negative examples


  for (Long64_t i = 0; i < nLuse; i++) {
    tL->GetEntry(i);

    // apply cuts
    if (edtmL > EDTM_TDC_MAX)
      continue;
    if (dpL < HDELTA_MIN || dpL > HDELTA_MAX)
      continue;
    if (etotL < HCALTOT_MIN)
      continue;
    if (npeL < HCERNPE_MIN)
      continue;
    
    int nL_i = (int)nclustL_d;
    if (nL_i < 0 || nL_i > nclustMax) continue;
    
    EventClusSummary sL = SummarizeEvent(nL_i, clusE_L, clusT_L,
                                        block_clusterID_L, goodAdcTdcDiffTime_L,
                                        CLUS_ECUT, CLUS_TMIN, CLUS_TMAX,
                                        DT_ABS_MAX);
    
    auto ins = looseEvt.emplace(evtNumL, sL);
    if (!ins.second) nLooseDup++;
    nLoosePass++;
  }

  Long64_t nT = tT->GetEntries();
  Long64_t nTuse = nT;
  if (maxEntries > 0 && maxEntries < nTuse)
    nTuse = maxEntries;

  for (Long64_t i = 0; i < nTuse; i++) {
    tT->GetEntry(i);

    if (edtmT > EDTM_TDC_MAX)
      continue;
    if (dpT < HDELTA_MIN || dpT > HDELTA_MAX)
      continue;
    if (etotT < HCALTOT_MIN)
      continue;
    if (npeT < HCERNPE_MIN)
      continue;

    int nT_i = (int)nclustT_d;
    if (nT_i < 0 || nT_i > nclustMax) continue;

    nTightPass++;
  
  auto it = looseEvt.find(evtNumT);
  if (it == looseEvt.end()) { nTightOnly++; continue; }
  
  EventClusSummary sL = it->second;
  
  EventClusSummary sT = SummarizeEvent(nT_i, clusE_T, clusT_T,
                                      block_clusterID_T, goodAdcTdcDiffTime_T,
                                      CLUS_ECUT, CLUS_TMIN, CLUS_TMAX,
                                      DT_ABS_MAX);
  
  // existing:
  int dNclust = nT_i - sL.nclust;
  hDelta->Fill(dNclust);
  hMig->Fill(sL.nclust, nT_i);
  
  // new (absolute deltas):
  hDeltaNblkAll->Fill(sT.nblk_all - sL.nblk_all);
  hDeltaNblkSel->Fill(sT.nblk_sel - sL.nblk_sel);
  hDeltaEsumSel->Fill(sT.Esum_sel - sL.Esum_sel);
  hDeltaEmaxSel->Fill(sT.Emax_sel - sL.Emax_sel);

  // new (fractional deltas, only if loose reference is meaningful):
  if (sL.Esum_sel > EREF_MIN) {
    hFracEsumSel->Fill((sT.Esum_sel - sL.Esum_sel) / sL.Esum_sel);
  }
  if (sL.Emax_sel > EREF_MIN) {
    hFracEmaxSel->Fill((sT.Emax_sel - sL.Emax_sel) / sL.Emax_sel);
  }

  nMatchedPass++;
  if (sL.nclust == nT_i) nSame++;
  }

  std::cout << "Loose pass (after cuts): " << nLoosePass << "\n";
  std::cout << "Loose duplicates (evtNum collisions): " << nLooseDup << "\n";
  std::cout << "Tight pass (after cuts): " << nTightPass << "\n";
  std::cout << "Matched pass (both, by evtNum): " << nMatchedPass << "\n";
  std::cout << "Tight-only (no loose match): " << nTightOnly << "\n";

  std::cout << "Delta summary: min=" << minDelta
            << " max=" << maxDelta
            << " nNeg=" << nNegDelta << "\n";

  // ---- migration diagnostics: any content below diagonal? (tight < loose)
  long long nBelowDiag = 0;
  for (int x = 0; x <= nclustMax; x++) {
    for (int y = 0; y < x; y++) { // y < x => below diagonal (tight smaller)
      long long c = (long long)hMig->GetBinContent(x + 1, y + 1);
      if (c > 0) {
        nBelowDiag += c;
        std::cout << "Below diag cell: loose=" << x
                  << " tight=" << y
                  << " count=" << c << "\n";
      }
    }
  }
  std::cout << "Total below-diagonal counts = " << nBelowDiag << "\n";

  // ===============================
  // CSV OUTPUT (one row per nonzero migration cell)
  // ===============================
  std::string csvPath = "csv/clus_migration_stats.csv";
  EnsureCSVHeader(csvPath);

  for (int x = 0; x <= nclustMax; x++) {
    for (int y = 0; y <= nclustMax; y++) {
      long long c = (long long)hMig->GetBinContent(x + 1, y + 1);
      if (c == 0)
        continue;

      AppendCSVRow(csvPath, run, seg, nsLoose, nsTight, maxEntries,
                   EDTM_TDC_MAX, HDELTA_MIN, HDELTA_MAX, HCALTOT_MIN,
                   HCERNPE_MIN, nMatchedPass, nSame, x, y, c);
    }
  }

  // ===============================
  // DRAW + SAVE
  // ===============================
 {
  TCanvas c1("c_mig", "", 1050, 850);

  // --- two pads: header (title+stamp) + main (plot)
  TPad *pHead = new TPad("pHead_mig", "pHead_mig", 0.0, 0.84, 1.0, 1.0);
  TPad *pMain = new TPad("pMain_mig", "pMain_mig", 0.0, 0.0, 1.0, 0.84);

  pHead->SetFillStyle(0);
  pHead->SetBorderMode(0);
  pHead->SetBottomMargin(0.0);
  pHead->SetTopMargin(0.25);
  pHead->SetLeftMargin(0.12);
  pHead->SetRightMargin(0.06);

  pMain->SetFillStyle(0);
  pMain->SetBorderMode(0);
  pMain->SetTopMargin(0.02);
  pMain->SetBottomMargin(0.12);
  pMain->SetLeftMargin(0.12);
  pMain->SetRightMargin(0.10); // a bit more room for the COLZ palette

  pHead->Draw();
  pMain->Draw();

  // ---- MAIN plot pad
  pMain->cd();
  gPad->SetTicks(1, 1);

  // kill ROOT title so it doesn't fight the header
  hMig->SetTitle("");

  hMig->Draw("COLZ TEXT");
  gPad->Update();

  // ---- HEADER pad: title + stamp
  pHead->cd();
  {
    TLatex lat;
    lat.SetNDC(true);

    // ---- title (centered)
    lat.SetTextAlign(23);
    lat.SetTextSize(0.26);
    lat.DrawLatex(0.5, 0.95, "nclust migration");

    // ---- stamp (match your delta style)
    lat.SetTextAlign(11);
    lat.SetTextSize(0.15);

    double x = 0.12;
    double y = 0.55;
    double dy = 0.20;

    lat.DrawLatex(x, y,
                  Form("Run %d | seg %d | nclust migration: %d ns #rightarrow %d ns",
                       run, seg, nsLoose, nsTight));
    y -= dy;
    lat.DrawLatex(x, y,
                  Form("Cuts: EDTM<%.2f | dp[%.0f,%.0f] | etot>%.2f | npe>%.1f",
                       EDTM_TDC_MAX, HDELTA_MIN, HDELTA_MAX, HCALTOT_MIN, HCERNPE_MIN));
    y -= dy;

    double fracSame = (nMatchedPass > 0) ? double(nSame) / double(nMatchedPass) : 0.0;
    lat.DrawLatex(x, y,
                  Form("Events passing cuts in BOTH trees: Total %lld | unchanged %lld | frac %.4f",
                       (long long)nMatchedPass, (long long)nSame, fracSame));
  }

  c1.cd();
  SaveCanvasPNG(&c1, run, seg, nsLoose, nsTight, "mig_nclust");
} 

{
  TCanvas c2("c_delta", "", 900, 700);

  // --- two pads: header (stamp) + main (plot)
  
  TPad *pHead = new TPad("pHead", "pHead", 0.0, 0.84, 1.0, 1.0);
  TPad *pMain = new TPad("pMain", "pMain", 0.0, 0.0, 1.0, 0.84);
  
  pHead->SetFillStyle(0);
  pHead->SetBorderMode(0);
  pHead->SetBottomMargin(0.0);
  pHead->SetTopMargin(0.25);
  pHead->SetLeftMargin(0.12);
  pHead->SetRightMargin(0.06);

  pMain->SetFillStyle(0);
  pMain->SetBorderMode(0);
  pMain->SetTopMargin(0.02);
  pMain->SetBottomMargin(0.12);
  pMain->SetLeftMargin(0.12);
  pMain->SetRightMargin(0.06);

  pHead->Draw();
  pMain->Draw();

  // ---- MAIN plot pad
  pMain->cd();
  gPad->SetLogy();
  gPad->SetTicks(1, 0);

  hDelta->SetLineWidth(2);
  hDelta->SetMinimum(0.5);
  hDelta->GetXaxis()->SetRangeUser(-10, 15);

  TAxis *ax = hDelta->GetXaxis();
  ax->SetLabelSize(0.035);
  ax->SetTitleSize(0.04);
  ax->SetTitleOffset(1.1);
  ax->SetNdivisions(510);

  hDelta->Draw("HIST");
  gPad->Update();

  // ---- HEADER pad: draw ONLY the stamp (and no title in the plot)
  pHead->cd();

  // disable histogram title so it doesn't compete with stamp
  hDelta->SetTitle("");

  // draw stamp in header pad coordinates
  {
    TLatex lat;
    lat.SetNDC(true);

    // ---- title (centered)
    lat.SetTextAlign(23);      // center, top
    lat.SetTextSize(0.26);
    lat.DrawLatex(0.5, 0.95, "#Delta N_{clust} (tight - loose)");

    // ---- stamp (right aligned)
    lat.SetTextAlign(11);  //left, top
    lat.SetTextSize(0.15);   //larger ~ 0.22
    double x = 0.12;  // aligh with plot left margin
    double y = 0.55;  // start below the title
    double dy = 0.20; //line spacing

    lat.DrawLatex(x, y,
                  Form("Run %d | seg %d | nclust migration: %d ns #rightarrow %d ns",
                       run, seg, nsLoose, nsTight));
    y -= dy;
    lat.DrawLatex(x, y,
                  Form("Cuts: EDTM<%.2f | dp[%.0f,%.0f] | etot>%.2f | npe>%.1f",
                       EDTM_TDC_MAX, HDELTA_MIN, HDELTA_MAX, HCALTOT_MIN, HCERNPE_MIN));
    y -= dy;

    double fracSame = (nMatchedPass > 0) ? double(nSame) / double(nMatchedPass) : 0.0;
    lat.DrawLatex(x, y,
                  Form("Events passing cuts in BOTH trees: Total %lld | unchanged %lld | frac %.4f",
                       (long long)nMatchedPass, (long long)nSame, fracSame));
  }

  c2.cd();
  SaveCanvasPNG(&c2, run, seg, nsLoose, nsTight, "delta_nclust");
}


  // ---- additional event-by-event deltas
  DrawAndSaveDelta1D(hDeltaNblkAll, run, seg, nsLoose, nsTight,
                     "#Delta N_{blk} all (tight - loose)", "delta_nblk_all",
                     EDTM_TDC_MAX, HDELTA_MIN, HDELTA_MAX, HCALTOT_MIN, HCERNPE_MIN,
                     nMatchedPass, nSame,
                     true, false, 0, 0);

  DrawAndSaveDelta1D(hDeltaNblkSel, run, seg, nsLoose, nsTight,
                     "#Delta N_{blk} selected (tight - loose)", "delta_nblk_sel",
                     EDTM_TDC_MAX, HDELTA_MIN, HDELTA_MAX, HCALTOT_MIN, HCERNPE_MIN,
                     nMatchedPass, nSame,
                     true, false, 0, 0);

  DrawAndSaveDelta1D(hDeltaEsumSel, run, seg, nsLoose, nsTight,
                     "#Delta E_{sum} selected (tight - loose)", "delta_esum_sel",
                     EDTM_TDC_MAX, HDELTA_MIN, HDELTA_MAX, HCALTOT_MIN, HCERNPE_MIN,
                     nMatchedPass, nSame,
                     true, false, 0, 0);

  DrawAndSaveDelta1D(hDeltaEmaxSel, run, seg, nsLoose, nsTight,
                     "#Delta E_{max} selected (tight - loose)", "delta_emax_sel",
                     EDTM_TDC_MAX, HDELTA_MIN, HDELTA_MAX, HCALTOT_MIN, HCERNPE_MIN,
                     nMatchedPass, nSame,
                     true, false, 0, 0);
  
  DrawAndSaveDelta1D(hFracEsumSel, run, seg, nsLoose, nsTight,
                     "Frac #Delta E_{sum} selected (tight - loose) / loose", "frac_esum_sel",
                     EDTM_TDC_MAX, HDELTA_MIN, HDELTA_MAX, HCALTOT_MIN, HCERNPE_MIN,
                     nMatchedPass, nSame,
                     true, false, 0, 0);

  DrawAndSaveDelta1D(hFracEmaxSel, run, seg, nsLoose, nsTight,
                     "Frac #Delta E_{max} selected (tight - loose) / loose", "frac_emax_sel",
                     EDTM_TDC_MAX, HDELTA_MIN, HDELTA_MAX, HCALTOT_MIN, HCERNPE_MIN,
                     nMatchedPass, nSame,
                     true, false, 0, 0);

  delete hMig;
  delete hDelta;
  delete hDeltaNblkAll;
  delete hDeltaNblkSel;
  delete hDeltaEsumSel;
  delete hDeltaEmaxSel;
  delete hFracEsumSel;
  delete hFracEmaxSel;

  fL->Close();
  fT->Close();
  delete fL;
  delete fT;

  std::cout << "Done.\n";
}
