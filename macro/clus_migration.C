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

#include "TCanvas.h"
#include "TFile.h"
#include "TH1I.h"
#include "TH2I.h"
#include "TLatex.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"

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
// Plot stamp
// -------------------------------
static void StampMigration(int run,int seg, int nsLoose,
                           int nsTight, double edtmtdc_max, double hdelta_min,
                           double hdelta_max, double hcaltot_min,
                           double hcernpe_min, Long64_t maxEntries,
                           Long64_t nPass, Long64_t nSame) {
  TLatex lat;
  lat.SetNDC(true);
  lat.SetTextAlign(31);   // right-aligned, vertically centered
  lat.SetTextSize(0.028); // smaller, non-dominant

  double x = 0.95;
  double y = 0.88;
  double dy = 0.045;

  lat.DrawLatex(
      x, y,
      Form("Run %d | seg %d | nclust migration: %d ns #rightarrow %d ns", run,
           seg, nsLoose, nsTight));

  y -= dy;
  lat.DrawLatex(x, y,
                Form("Cuts: EDTM<%.2f | dp[%.0f,%.0f] | etot>%.2f | npe>%.1f | "
                     "maxEntries=%lld",
                     edtmtdc_max, hdelta_min, hdelta_max, hcaltot_min,
                     hcernpe_min, maxEntries));

  y -= dy;
  lat.DrawLatex(
      x, y,
      Form("Events passing cuts in BOTH trees: %lld | unchanged: %lld", nPass,
           nSame));
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

  tL->SetBranchStatus("fEvtHdr.fEvtNum", 1);
  tT->SetBranchStatus("fEvtHdr.fEvtNum", 1);

  tL->SetBranchStatus("T.hms.hEDTM_tdcTimeRaw", 1);
  tL->SetBranchStatus("H.gtr.dp", 1);
  tL->SetBranchStatus("H.cal.etotnorm", 1);
  tL->SetBranchStatus("H.cer.npeSum", 1);
  tL->SetBranchStatus("NPS.cal.nclust", 1);

  tT->SetBranchStatus("T.hms.hEDTM_tdcTimeRaw", 1);
  tT->SetBranchStatus("H.gtr.dp", 1);
  tT->SetBranchStatus("H.cal.etotnorm", 1);
  tT->SetBranchStatus("H.cer.npeSum", 1);
  tT->SetBranchStatus("NPS.cal.nclust", 1);

  // ---- set addresses
  double edtmL = 0, dpL = 0, etotL = 0, npeL = 0, nclustL_d = 0;
  double edtmT = 0, dpT = 0, etotT = 0, npeT = 0, nclustT_d = 0;

  UInt_t evtNumL = 0, evtNumT = 0;

  tL->SetBranchAddress("fEvtHdr.fEvtNum", &evtNumL);
  tT->SetBranchAddress("fEvtHdr.fEvtNum", &evtNumT);

  tL->SetBranchAddress("T.hms.hEDTM_tdcTimeRaw", &edtmL);
  tL->SetBranchAddress("H.gtr.dp", &dpL);
  tL->SetBranchAddress("H.cal.etotnorm", &etotL);
  tL->SetBranchAddress("H.cer.npeSum", &npeL);
  tL->SetBranchAddress("NPS.cal.nclust", &nclustL_d);

  tT->SetBranchAddress("T.hms.hEDTM_tdcTimeRaw", &edtmT);
  tT->SetBranchAddress("H.gtr.dp", &dpT);
  tT->SetBranchAddress("H.cal.etotnorm", &etotT);
  tT->SetBranchAddress("H.cer.npeSum", &npeT);
  tT->SetBranchAddress("NPS.cal.nclust", &nclustT_d);

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

  // ===============================
  // EVENT LOOP
  // building a lookup table to match the same physics event across two files
  // even when entry indices don’t line up
  // ===============================

  // evtNum -> nclustLoose (after cuts, clamped)
  std::unordered_map<UInt_t, int>
      looseNclust; // create lookup table aka hash map
  looseNclust.reserve((size_t)std::min<Long64_t>(
      tL->GetEntries(),
      2000000)); // don't let it resize while filling because speed

  Long64_t nL = tL->GetEntries();
  Long64_t nLuse = nL;
  if (maxEntries > 0 && maxEntries < nLuse)
    nLuse = maxEntries; // use all entries

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
    if (nL_i <0 || nL_i > nclustMax) continue;

    // insert; if already present, count duplicates
    // this stores the event keyed by event number
    auto ins = looseNclust.emplace(evtNumL, nL_i); // this keeps the first value and ignores the later ones
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

    auto it = looseNclust.find(evtNumT);
    if (it == looseNclust.end()) {
      nTightOnly++;
      continue;
    }

    int nL_i = it->second;

    int d = nT_i - nL_i;

    if (d < minDelta) minDelta = d;
    if (d > maxDelta) maxDelta = d;

    if (d < 0) {
      nNegDelta++;
      if (nPrintNeg < 20) {
        std::cout << "NEG d: evt=" << evtNumT
                  << " nL=" << nL_i
                  << " nT=" << nT_i
                  << " d=" << d << "\n";
        nPrintNeg++;
      }
    }

    hMig->Fill(nL_i, nT_i);
    hDelta->Fill(d);

    nMatchedPass++;
    if (nL_i == nT_i)
      nSame++;
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
    c1.cd();
    hMig->Draw("COLZ TEXT");
    StampMigration(run, seg, nsLoose, nsTight, EDTM_TDC_MAX, HDELTA_MIN,
                   HDELTA_MAX, HCALTOT_MIN, HCERNPE_MIN, maxEntries, nMatchedPass,
                   nSame);
    SaveCanvasPNG(&c1, run, seg, nsLoose, nsTight, "mig_nclust");
  }

{
  TCanvas c2("c_delta", "", 900, 700);
  c2.cd();

  gPad->SetTicks(1, 0);

  hDelta->SetLineWidth(2);

  TAxis *ax = hDelta->GetXaxis();

  ax->SetNdivisions(-NBINS_D);   // NEGATIVE = force exact bin-based ticks
  ax->SetLabelSize(0.0);         // turn OFF ROOT labels completely
  ax->SetTickLength(0.03);
  ax->SetNoExponent(true);

std::cout << "hDelta visible integral: " << hDelta->Integral() << "\n";
std::cout << "hDelta underflow (d<" << DMIN << "): " << hDelta->GetBinContent(0) << "\n";
std::cout << "hDelta overflow  (d>" << DMAX << "): " << hDelta->GetBinContent(NBINS_D + 1) << "\n";

  hDelta->Draw("HIST");
  gPad->Update();

  // ---- manual labels in DATA coordinates (not NDC)
  TLatex lab;
  lab.SetTextAlign(23);   // centered, top-aligned
  lab.SetTextSize(0.035);

  double y = gPad->GetUymin() - 0.04*(gPad->GetUymax()-gPad->GetUymin());

  int labels[] = {-10, -5, 0, 5, 10};
  for (int v : labels) {
    lab.DrawLatex(v, y, Form("%d", v));
  }

  StampMigration(run, seg, nsLoose, nsTight,
                 EDTM_TDC_MAX, HDELTA_MIN, HDELTA_MAX,
                 HCALTOT_MIN, HCERNPE_MIN,
                 maxEntries, nMatchedPass, nSame);

  SaveCanvasPNG(&c2, run, seg, nsLoose, nsTight, "delta_nclust");
}

  delete hMig;
  delete hDelta;

  fL->Close();
  fT->Close();
  delete fL;
  delete fT;

  std::cout << "Done.\n";
}
