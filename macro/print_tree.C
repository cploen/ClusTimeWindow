void print_tree(){
TFile *f = TFile::Open("rootfiles/nps_hms_coin_5sec_3013_0_1_-1.root");
if (!f || f->IsZombie()) return;
TTree *T = (TTree*)f->Get("T");
if (!T) return;
T->Print();
}
