// clus_basic_hists.C
// ROOT macro: make basic cluster histos + fits + CSV logging
//
// Run examples:
//   root -l 'clus_basic_hists.C(3013,5,0)'
//   root -l 'clus_basic_hists.C(3013,12,3,130,170,0.6,50000)'
//
// Output:
//   PNGs -> img/img_jan2026
//   CSV  -> csv/clus_timewindow_stats.csv

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TF1.h"
#include "TFile.h"
#include "TH1F.h"
#include "TLatex.h"
#include "TMath.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"

#include "TPad.h"
#include "TPaveStats.h"

static const int ROWS = 36;
static const int COLS = 30;
static const int N_BLOCKS = ROWS * COLS;

// -------------------------------
// Input path helper
// -------------------------------
static std::string BuildInputPath(int run, int winSec, int seg, int tag1 = 1,
                                  int tag2 = -1) {
  return Form("rootfiles/nps_hms_coin_%dsec_%d_%d_%d_%d.root", winSec, run, seg,
              tag1, tag2);
}

// -------------------------------
// Plot stamp
// -------------------------------
static void StampRunAndWindow(int run, int winSec, double tmin, double tmax,
                              double ecut) {
  TLatex lat;
  lat.SetNDC(true);
  lat.SetTextSize(0.035);
  lat.DrawLatex(0.12, 0.86,
                Form("Run %d | ClusTimeWin: %d s | clusT [%.0f, %.0f] ns | "
                     "clusE #geq %.2f",
                     run, winSec, tmin, tmax, ecut));
}

// -------------------------------
// Save canvas
// -------------------------------
static void SaveCanvasPNG(TCanvas *c, int run, int winSec, const char *hname) {
  std::string out = Form("img/img_jan2026/%d_%dsec_%s.png", run, winSec, hname);
  c->SaveAs(out.c_str());
  std::cout << "Saved: " << out << "\n";
}

// ===============================
// FIT + CSV INFRASTRUCTURE
// ===============================
struct FitSummary {
  bool ok = false;
  double mean = NAN, meanErr = NAN;
  double sigma = NAN, sigmaErr = NAN;
  double chi2ndf = NAN;
  double entries = 0;
  double fitMin = NAN, fitMax = NAN;
};

static void EnsureCSVHeader(const std::string &csvPath) {
  std::ifstream fin(csvPath);
  bool needHeader =
      (!fin.good()) || fin.peek() == std::ifstream::traits_type::eof();
  fin.close();

  if (needHeader) {
    std::ofstream fout(csvPath, std::ios::app);
    fout << "run,winSec,seg,observable,"
            "clusTmin,clusTmax,clusEcut,maxEntries,"
            "edtmtdc_max,hdelta_min,hdelta_max,hcaltot_min,hcernpe_min,"
            "fitModel,fitMin,fitMax,"
            "entries,mean,meanErr,sigma,sigmaErr,chi2ndf\n";
  }
}
static FitSummary FitGausAroundPeak(TH1 *h, double halfWidthX) {
  FitSummary s;
  if (!h)
    return s;
  s.entries = h->GetEntries();
  if (s.entries < 50)
    return s;

  int binMax = h->GetMaximumBin();
  double x0 = h->GetXaxis()->GetBinCenter(binMax);

  double fitMin = x0 - halfWidthX;
  double fitMax = x0 + halfWidthX;

  // clamp to axis
  fitMin = std::max(fitMin, h->GetXaxis()->GetXmin());
  fitMax = std::min(fitMax, h->GetXaxis()->GetXmax());

  // record the actual fit window used
  s.fitMin = fitMin;
  s.fitMax = fitMax;

  TF1 *f = new TF1(Form("f_%s_peakgaus", h->GetName()), "gaus", fitMin, fitMax);
  f->SetParameters(h->GetMaximum(), x0, h->GetRMS() > 0 ? h->GetRMS() : 1.0);

  int status = h->Fit(f, "RQ0");
  if (status != 0) {
    delete f;
    return s;
  }

  s.ok = true;
  s.mean = f->GetParameter(1);
  s.meanErr = f->GetParError(1);
  s.sigma = f->GetParameter(2);
  s.sigmaErr = f->GetParError(2);
  double ndf = f->GetNDF();
  s.chi2ndf = (ndf > 0) ? f->GetChisquare() / ndf : NAN;

  f->SetLineWidth(2);
  h->GetListOfFunctions()->Add(f);
  return s;
}

static void AppendCSVRow(const std::string &csvPath, int run, int winSec,
                         int seg, const std::string &obs, double clusTmin,
                         double clusTmax, double clusEcut, long long maxEntries,
                         double edtmtdc_max, double hdelta_min,
                         double hdelta_max, double hcaltot_min,
                         double hcernpe_min, const std::string &model,
                         const FitSummary &s) {
  std::ofstream fout(csvPath, std::ios::app);
  fout << run << "," << winSec << "," << seg << "," << obs << "," << clusTmin
       << "," << clusTmax << "," << clusEcut << "," << maxEntries << ","
       << edtmtdc_max << "," << hdelta_min << "," << hdelta_max << ","
       << hcaltot_min << "," << hcernpe_min << "," << model << "," << s.fitMin
       << "," << s.fitMax << "," << (long long)s.entries << ","
       << std::setprecision(8) << s.mean << "," << s.meanErr << "," << s.sigma
       << "," << s.sigmaErr << "," << s.chi2ndf << "\n";
}

// ===============================
// MAIN MACRO
// ===============================
void clus_basic_hists(int run = 3013, int winSec = 5, int seg = 0,
                      double clusTmin = 130.0, double clusTmax = 170.0,
                      double clusEcut = 0.60, Long64_t maxEntries = -1,
                      bool batch = true, bool verbose = false) {
  if (batch)
    gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(1110);

  // ---- event-level cuts (EXPLICIT + LOGGED)
  const double EDTM_TDC_MAX = 0.1;
  const double HDELTA_MIN = -12.0;
  const double HDELTA_MAX = 12.0;
  const double HCALTOT_MIN = 0.6;
  const double HCERNPE_MIN = 1.0;

  // ---- open file
  std::string inpath = BuildInputPath(run, winSec, seg);
  std::cout << "Opening: " << inpath << "\n";

  TFile *f = TFile::Open(inpath.c_str(), "READ");
  if (!f || f->IsZombie())
    return;

  TTree *t = (TTree *)f->Get("T");
  if (!t)
    return;

  // ===============================
  // BRANCH SETUP
  // ===============================

  t->SetBranchStatus("*", 0);

  // Enable only what is used downstream
  t->SetBranchStatus("T.hms.hEDTM_tdcTimeRaw", 1);
  t->SetBranchStatus("H.gtr.dp", 1);
  t->SetBranchStatus("H.cal.etotnorm", 1);
  t->SetBranchStatus("H.cer.npeSum", 1);

  t->SetBranchStatus("NPS.cal.nclust", 1);
  t->SetBranchStatus("NPS.cal.clusE", 1);
  t->SetBranchStatus("NPS.cal.clusT", 1);

  t->SetBranchStatus("NPS.cal.fly.block_clusterID", 1);
  t->SetBranchStatus("NPS.cal.fly.goodAdcTdcDiffTime", 1);

  // ---- branches (NOTE: TRIG6/TRIG1 removed)
  double edtmtdc = 0;
  double hdelta = 0, hcaltot = 0, hcernpe = 0;

  double block_clusterID[N_BLOCKS];
  double goodAdcTdcDiffTime[N_BLOCKS];

  double nclustDouble = 0;
  double clusE[10000];
  double clusT[10000];

  Long64_t nEntries = t->GetEntries();
  if (maxEntries > 0 && maxEntries < nEntries)
    nEntries = maxEntries;

  // ---- set addresses
  t->SetBranchAddress("T.hms.hEDTM_tdcTimeRaw", &edtmtdc);
  t->SetBranchAddress("H.gtr.dp", &hdelta);
  t->SetBranchAddress("H.cal.etotnorm", &hcaltot);
  t->SetBranchAddress("H.cer.npeSum", &hcernpe);

  t->SetBranchAddress("NPS.cal.fly.block_clusterID",
                      block_clusterID); // <-- no &
  t->SetBranchAddress("NPS.cal.fly.goodAdcTdcDiffTime",
                      goodAdcTdcDiffTime); // <-- no &

  t->SetBranchAddress("NPS.cal.nclust", &nclustDouble);
  t->SetBranchAddress("NPS.cal.clusE", clusE); // <-- no &
  t->SetBranchAddress("NPS.cal.clusT", clusT); // <-- no &

  // Build histograms
  // Number of clusters per event
  TH1F *h_nclust =
      new TH1F("h_nclust", "Clusters per event;N_{clust};Events", 40, 0, 40);

  // Cluster size (timing-qualified blocks) for clusters passing clusE/clusT
  // selection
  TH1F *h_clusSize = new TH1F(
      "h_clusSize",
      "Cluster size (timing-qualified blocks);N_{blocks in cluster};Clusters",
      60, 0, 60);

  // Block timing distribution for blocks in selected clusters
  TH1F *h_dtBlock =
      new TH1F("h_dtBlock",
               "Block dt (goodAdcTdcDiffTime) in selected clusters;dt;Blocks",
               200, clusTmin - 40, clusTmax + 40);

  // Cluster time distribution for selected clusters
  TH1F *h_clusT = new TH1F(
      "h_clusT", "Cluster time clusT for selected clusters;clusT;Clusters", 200,
      clusTmin - 40, clusTmax + 40);

  // Residual = block dt - clusT (resolution-like)
  TH1F *h_resid = new TH1F(
      "h_resid",
      "Timing residual: dt - clusT (selected clusters);dt - clusT;Blocks", 240,
      -60, 60);

  // Selected cluster energy
  TH1F *h_clusE = new TH1F(
      "h_clusE", "Cluster energy clusE for selected clusters;clusE;Clusters",
      200, 0, 4.0);

  // ---- loop
  for (Long64_t ev = 0; ev < nEntries; ev++) {
    t->GetEntry(ev);

    if (edtmtdc > EDTM_TDC_MAX)
      continue;
    if (hdelta < HDELTA_MIN || hdelta > HDELTA_MAX)
      continue;
    if (hcaltot < HCALTOT_MIN)
      continue;
    if (hcernpe < HCERNPE_MIN)
      continue;

    int nclust = (int)nclustDouble;
    h_nclust->Fill(nclust);

    std::vector<char> clusSel(nclust, 0);
    for (int cid = 0; cid < nclust; cid++) {
      double e = clusE[cid];
      double tclus = clusT[cid];

      if (e >= clusEcut && tclus >= clusTmin && tclus <= clusTmax) {
        clusSel[cid] = 1;

        // ✅ fill cluster-level distributions
        h_clusT->Fill(tclus);
        h_clusE->Fill(e);
      }
    }

    std::vector<int> size(nclust, 0);
    const double DT_ABS_MAX = 1e6;

    for (int ib = 0; ib < N_BLOCKS; ib++) {
      int cid = (int)block_clusterID[ib];
      if (cid < 0 || cid >= nclust)
        continue;

      double dt = goodAdcTdcDiffTime[ib];
      if (!std::isfinite(dt))
        continue;
      if (std::fabs(dt) > DT_ABS_MAX)
        continue;

      // count timing-qualified blocks per cluster
      size[cid]++;

      // ✅ only fill timing plots for selected clusters
      if (clusSel[cid]) {
        h_dtBlock->Fill(dt);
        h_resid->Fill(dt - clusT[cid]);
      }
    }

    for (int cid = 0; cid < nclust; cid++) {
      if (clusSel[cid])
        h_clusSize->Fill(size[cid]);
    }
  }

  // ---- FITS + CSV
  std::string csvPath = "csv/clus_timewindow_stats.csv";
  EnsureCSVHeader(csvPath);

  FitSummary fs_nclust =
      FitGausAroundPeak(h_nclust, 2.0); // half-width in x-units
  FitSummary fs_clusSize = FitGausAroundPeak(h_clusSize, 2.0);

  AppendCSVRow(csvPath, run, winSec, seg, "nclust_per_event", clusTmin,
               clusTmax, clusEcut, maxEntries, EDTM_TDC_MAX, HDELTA_MIN,
               HDELTA_MAX, HCALTOT_MIN, HCERNPE_MIN, "gaus", fs_nclust);

  AppendCSVRow(csvPath, run, winSec, seg, "cluster_size", clusTmin, clusTmax,
               clusEcut, maxEntries, EDTM_TDC_MAX, HDELTA_MIN, HDELTA_MAX,
               HCALTOT_MIN, HCERNPE_MIN, "gaus", fs_clusSize);

  // ---- draw + save
  auto draw_and_save = [&](TH1 *h) {
    TCanvas c("c", "", 900, 700);
    c.cd();

    h->SetLineWidth(2);
    h->Draw("HIST");

    // draw any attached fit functions on top
    if (h->GetListOfFunctions())
      h->GetListOfFunctions()->Draw("same");

    // must update before "stats" exists
    gPad->Update();

    // move stats box (per-plot)
    TPaveStats *st = (TPaveStats *)h->FindObject("stats");
    if (st) {
      st->SetX1NDC(0.70);
      st->SetX2NDC(0.90);
      st->SetY1NDC(0.60);
      st->SetY2NDC(0.78);
    }

    // stamp run/window info
    StampRunAndWindow(run, winSec, clusTmin, clusTmax, clusEcut);

    gPad->Modified();
    gPad->Update();

    SaveCanvasPNG(&c, run, winSec, h->GetName());
  };

  draw_and_save(h_nclust);
  draw_and_save(h_clusSize);
  draw_and_save(h_dtBlock);
  draw_and_save(h_clusT);
  draw_and_save(h_resid);
  draw_and_save(h_clusE);

  delete h_nclust;
  delete h_clusSize;
  delete h_dtBlock;
  delete h_clusT;
  delete h_resid;
  delete h_clusE;

  f->Close();
  delete f;

  std::cout << "Done.\n";
}
