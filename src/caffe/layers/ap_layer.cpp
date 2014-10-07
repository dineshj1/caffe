#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void APLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    //could get modes etc. for future, for small variations in computing AP
}

template <typename Dtype>
void APLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The scores and label should have the same number.";//and this should be equal to 2?
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1); // will hold AP
  (*top)[1]->Reshape(1, 1, 1, 1); // will hold ROC 
}

template <typename Dtype>
void APLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_scores = bottom[0]->cpu_data();// intended for now to be in the form of distances i.e. lower score => more likely to be positive
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int dim = bottom[0]->count()/bottom[0]->num(); //number of data points, in this case equal to the number of elements 
  // number of positives and negatives
  double P=0, N=0;

  for(int k=0; k<dim; ++k){
      if(bottom_label[k]==1){
          P=P+1;
      }else if(bottom_label[k]==0){
          N=N+1;
      }else{
          LOG(FATAL) << "Unknown label" << bottom_label;
      }
  }

  //sorting 
  std::vector<std::pair<Dtype, int> > bottom_data_vector;
  for (int j = 0; j < dim; ++j) {
     bottom_data_vector.push_back(std::make_pair(bottom_scores[j], bottom_label[j]));//pairs of scores and labels
  }
  std::sort(bottom_data_vector.begin(), bottom_data_vector.end()); //sorts in ascending order 

  //setting up variables to be updated in loop
  double FP=0, TP=0, FP_prev=0, TP_prev=0;
  double Prec=0, Rec=0, Prec_prev=0, Rec_prev=0;
  Dtype AP = 0, AUROC =0;
  Dtype score_prev=-1; // impossible to have negative values
  Dtype curr_score, curr_label;

  for(int k=0;k<dim; ++k){
    curr_score=bottom_data_vector[k].first;
    curr_label=bottom_data_vector[k].second;
    if(curr_score!=score_prev){// finalize numbers for the previous threshold
        AUROC=AUROC+0.5*abs(FP_prev-FP)*(TP_prev+TP);//trapezoid area
        AP=AP+0.5*abs(Rec_prev-Rec)*(Prec_prev+Prec);//trapezoid area
        score_prev=curr_score;
        FP_prev=FP;
        TP_prev=TP;
        Prec_prev=Prec;
        Rec_prev=Rec;
    }
    if(curr_label==1){
        TP=TP+1;
        //P=P+1;  //number of positives
    }else if(curr_label==0){
        FP=FP+1;
        //N=N+1;  //number of negatives
    }else{
        LOG(FATAL)<<"Unknown label "<< curr_label;
    }
  }
  Prec=TP/(TP+FP);
  Rec=TP/P;
  AP = AP+0.5*abs(1-Rec_prev)*(1+Prec_prev);//Last trapezoid area 
  AUROC=AUROC+0.5*abs(1-FP_prev)*(1+TP_prev);//Last trapezoid area
  AUROC=AUROC/(P*N);

  LOG(INFO) << "AP" << AP;
  (*top)[1]->mutable_cpu_data()[0] = AP;
  LOG(INFO) << "AUROC" << AUROC;
  (*top)[1]->mutable_cpu_data()[1] = AUROC;
  // AP layer should not be used as a loss function.
}

INSTANTIATE_CLASS(APLayer);
}  // namespace caffe

