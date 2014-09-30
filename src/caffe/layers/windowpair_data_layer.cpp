#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > WindowPairDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
WindowPairDataLayer<Dtype>::~WindowPairDataLayer<Dtype>() { //destructor
  this->JoinPrefetchThread();
}

template <typename Dtype>
void WindowPairDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // LayerSetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    ground_img_path (abs path)
  //    sat_img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    overlap xg1 yg1 xg2 yg2 xs1 ys1 xs2 ys2

  LOG(INFO) << "Window Pair data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.window_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.window_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.window_data_param().fg_fraction();

  const bool prefetch_needs_rand =
      this->transform_param_.mirror() ||
      this->transform_param_.crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }

  std::ifstream infile(this->layer_param_.window_data_param().source().c_str());  //checking that file can be opened
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.window_data_param().source() << std::endl;

  map<int, int> label_hist; //number of windows assigned to each label (in this case, only similar/dissimilar - not explicitly given... just inferred) - key: classname, value: #windows
  label_hist.insert(std::make_pair(0, 0)); // 0 corresponds to be background class

  string hashtag;
  int image_index, channels;
  if (!(infile >> hashtag >> image_index)) { //getting first image index
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string ground_img_path, sat_img_path;
    infile >> ground_img_path >> sat_img_path; // get image filename
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2]; //get image dimensions - channels, height, width
    channels = image_size[0];
    imagepair_database_.push_back(std::make_pair(std::make_pair(ground_img_path, sat_img_path), image_size)); //database stored as map from image path to size

    // read each box
    int num_windows;
    infile >> num_windows;  //get number of windows
    const float fg_threshold =
        this->layer_param_.window_data_param().fg_threshold();
    const float bg_threshold =
        this->layer_param_.window_data_param().bg_threshold();
    for (int i = 0; i < num_windows; ++i) {// for each box
      int xg1, yg1, xg2, yg2, xs1, ys1, xs2, ys2;
      float overlap;
      infile >> overlap >> xg1 >> yg1 >> xg2 >> yg2 >> xs1 >> ys1 >> xs2 >> ys2;

      vector<float> window(WindowPairDataLayer::NUM);
      window[WindowPairDataLayer::IMAGE_INDEX] = image_index;
      window[WindowPairDataLayer::OVERLAP] = overlap;
      window[WindowPairDataLayer::XG1] = xg1;
      window[WindowPairDataLayer::YG1] = yg1;
      window[WindowPairDataLayer::XG2] = xg2;
      window[WindowPairDataLayer::YG2] = yg2;
      window[WindowPairDataLayer::XS1] = xs1;
      window[WindowPairDataLayer::YS1] = ys1;
      window[WindowPairDataLayer::XS2] = xs2;
      window[WindowPairDataLayer::YS2] = ys2;
   
      // add window to foreground list or background list
      if (overlap >= fg_threshold) {
        window[WindowPairDataLayer::LABEL] = 1;
        int label = window[WindowPairDataLayer::LABEL];
        CHECK_GT(label, 0); // no class is assigned 0 label
        fg_windows_.push_back(window);
        label_hist.insert(std::make_pair(1, 0));
        label_hist[1]++; 
      } else if (overlap < bg_threshold) {
        // background window, force label and overlap to 0
        window[WindowPairDataLayer::LABEL] = 0;
        window[WindowPairDataLayer::OVERLAP] = 0;
        bg_windows_.push_back(window);
        label_hist[0]++;
      }
    }

    if (image_index % 100 == 0) { // for every hundredth image
      LOG(INFO) << "num: " << image_index << " "
          << ground_img_path << " "
          << sat_img_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  } while (infile >> hashtag >> image_index); // till end of file

  LOG(INFO) << "Number of images: " << image_index+1;

  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {// only two classes - match and non-match
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.window_data_param().context_pad();

  LOG(INFO) << "Crop mode: "
      << this->layer_param_.window_data_param().crop_mode();

  // image
  const int crop_size = this->transform_param_.crop_size();
  CHECK_GT(crop_size, 0);
  const int batch_size = this->layer_param_.window_data_param().batch_size();
  (*top)[0]->Reshape(batch_size, channels, crop_size, crop_size);
  (*top)[1]->Reshape(batch_size, channels, crop_size, crop_size);
  this->prefetch_ground_.Reshape(batch_size, channels, crop_size, crop_size);
  this->prefetch_sat_.Reshape(batch_size, channels, crop_size, crop_size);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // datum size
  this->datum_channels_ = (*top)[0]->channels();
  this->datum_height_ = (*top)[0]->height();
  this->datum_width_ = (*top)[0]->width();
  this->datum_size_ =
      (*top)[0]->channels() * (*top)[0]->height() * (*top)[0]->width();
  // label
  (*top)[2]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1); //TODO: should this be omitted? Where are we setting this?
}

template <typename Dtype>
unsigned int WindowPairDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void WindowPairDataLayer<Dtype>::InternalThreadEntry() {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_ground = this->prefetch_ground_.mutable_cpu_data();
  Dtype* top_sat = this->prefetch_sat_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  const Dtype scale = this->layer_param_.window_data_param().scale();
  const int batch_size = this->layer_param_.window_data_param().batch_size();
  const int context_pad = this->layer_param_.window_data_param().context_pad();
  const int crop_size = this->transform_param_.crop_size();
  const bool mirror = this->transform_param_.mirror();
  const float fg_fraction =
      this->layer_param_.window_data_param().fg_fraction();
  const Dtype* mean = this->data_mean_.cpu_data(); //TODO: Is it possible to have two separate means for the the two input streams?
  const int mean_off = (this->data_mean_.width() - crop_size) / 2;
  const int mean_width = this->data_mean_.width();
  const int mean_height = this->data_mean_.height();
  cv::Size cv_crop_size(crop_size, crop_size);
  const string& crop_mode = this->layer_param_.window_data_param().crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  caffe_set(this->prefetch_ground_.count(), Dtype(0), top_ground);
  caffe_set(this->prefetch_sat_.count(), Dtype(0), top_sat);

  const int num_fg = static_cast<int>(static_cast<float>(batch_size)
      * fg_fraction);
  const int num_samples[2] = { batch_size - num_fg, num_fg };

  int item_id = 0;
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      // sample a window
      const unsigned int rand_index = PrefetchRand();
      vector<float> window = (is_fg) ?
          fg_windows_[rand_index % fg_windows_.size()] :
          bg_windows_[rand_index % bg_windows_.size()];

      bool do_mirror = false;
      if (mirror && PrefetchRand() % 2) {
        do_mirror = true;
      }

      // load the image containing the window
      pair<pair<std::string, std::string>, vector<int> > image =
          imagepair_database_[window[WindowPairDataLayer<Dtype>::IMAGE_INDEX]];

      std::string img_path[2];
      img_path[0]=image.first.first; // ground_img_path
      img_path[1]=image.first.second; // sat_img_path

      // crop window out of image and warp it
      int x1[2], y1[2], x2[2], y2[2];
      x1[0] = window[WindowPairDataLayer<Dtype>::XG1];
      y1[0] = window[WindowPairDataLayer<Dtype>::YG1];
      x2[0] = window[WindowPairDataLayer<Dtype>::XG2];
      y2[0] = window[WindowPairDataLayer<Dtype>::YG2];
      x1[1] = window[WindowPairDataLayer<Dtype>::XS1];
      y1[1] = window[WindowPairDataLayer<Dtype>::YS1];
      x2[1] = window[WindowPairDataLayer<Dtype>::XS2];
      y2[1] = window[WindowPairDataLayer<Dtype>::YS2];
       

      for(int imgno=0; imgno<2; imgno++){ // ground and sat images
        cv::Mat cv_img = cv::imread(img_path[imgno], CV_LOAD_IMAGE_COLOR);
        if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << img_path[imgno];
        return;
        } 
        const int channels = cv_img.channels();

        int pad_w = 0;
        int pad_h = 0;
        if (context_pad > 0 || use_square) {
          // scale factor by which to expand the original region
          // such that after warping the expanded region to crop_size x crop_size
          // there's exactly context_pad amount of padding on each side
          Dtype context_scale = static_cast<Dtype>(crop_size) /
              static_cast<Dtype>(crop_size - 2*context_pad);

          // compute the expanded region
          Dtype half_height = static_cast<Dtype>(y2[imgno]-y1[imgno]+1)/2.0;
          Dtype half_width = static_cast<Dtype>(x2[imgno]-x1[imgno]+1)/2.0;
          Dtype center_x = static_cast<Dtype>(x1[imgno]) + half_width;
          Dtype center_y = static_cast<Dtype>(y1[imgno]) + half_height;
          if (use_square) {
            if (half_height > half_width) {
              half_width = half_height;
            } else {
              half_height = half_width;
            }
          }
          x1[imgno] = static_cast<int>(round(center_x - half_width*context_scale));
          x2[imgno] = static_cast<int>(round(center_x + half_width*context_scale));
          y1[imgno] = static_cast<int>(round(center_y - half_height*context_scale));
          y2[imgno] = static_cast<int>(round(center_y + half_height*context_scale));

          // the expanded region may go outside of the image
          // so we compute the clipped (expanded) region and keep track of
          // the extent beyond the image
          int unclipped_height = y2[imgno]-y1[imgno]+1;
          int unclipped_width = x2[imgno]-x1[imgno]+1;
          int pad_x1 = std::max(0, -x1[imgno]);
          int pad_y1 = std::max(0, -y1[imgno]);
          int pad_x2 = std::max(0, x2[imgno] - cv_img.cols + 1);
          int pad_y2 = std::max(0, y2[imgno] - cv_img.rows + 1);
          // clip bounds
          x1[imgno] = x1[imgno] + pad_x1;
          x2[imgno] = x2[imgno] - pad_x2;
          y1[imgno] = y1[imgno] + pad_y1;
          y2[imgno] = y2[imgno] - pad_y2;
          CHECK_GT(x1[imgno], -1);
          CHECK_GT(y1[imgno], -1);
          CHECK_LT(x2[imgno], cv_img.cols);
          CHECK_LT(y2[imgno], cv_img.rows);

          int clipped_height = y2[imgno]-y1[imgno]+1;
          int clipped_width = x2[imgno]-x1[imgno]+1;

          // scale factors that would be used to warp the unclipped
          // expanded region
          Dtype scale_x =
              static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
          Dtype scale_y =
              static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

          // size to warp the clipped expanded region to
          cv_crop_size.width =
              static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
          cv_crop_size.height =
              static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
          pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
          pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
          pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
          pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

          pad_h = pad_y1;
          // if we're mirroring, we mirror the padding too (to be pedantic)
          if (do_mirror) {
            pad_w = pad_x2;
          } else {
            pad_w = pad_x1;
          }

          // ensure that the warped, clipped region plus the padding fits in the
          // crop_size x crop_size image (it might not due to rounding)
          if (pad_h + cv_crop_size.height > crop_size) {
            cv_crop_size.height = crop_size - pad_h;
          }
          if (pad_w + cv_crop_size.width > crop_size) {
            cv_crop_size.width = crop_size - pad_w;
          }
        }

        cv::Rect roi(x1[imgno], y1[imgno], x2[imgno]-x1[imgno]+1, y2[imgno]-y1[imgno]+1);
        cv::Mat cv_cropped_img = cv_img(roi);
        cv::resize(cv_cropped_img, cv_cropped_img,
            cv_crop_size, 0, 0, cv::INTER_LINEAR);

        // horizontal flip at random
        if (do_mirror) {
          cv::flip(cv_cropped_img, cv_cropped_img, 1);
        }

        // copy the warped window into top_data
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < cv_cropped_img.rows; ++h) {
            for (int w = 0; w < cv_cropped_img.cols; ++w) {
              Dtype pixel =
                  static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);

              if(imgno==0){
              top_ground[((item_id * channels + c) * crop_size + h + pad_h)
                       * crop_size + w + pad_w]
                  = (pixel
                      - mean[(c * mean_height + h + mean_off + pad_h)
                             * mean_width + w + mean_off + pad_w])
                    * scale;
              }else{
               top_sat[((item_id * channels + c) * crop_size + h + pad_h)
                       * crop_size + w + pad_w]
                  = (pixel
                      - mean[(c * mean_height + h + mean_off + pad_h)
                             * mean_width + w + mean_off + pad_w])
                    * scale;
              }
            }
          }
        }
      }

      // get window label
      top_label[item_id] = window[WindowPairDataLayer<Dtype>::LABEL];

      #if 0
      // useful debugging code for dumping transformed windows to disk
      string file_id;
      std::stringstream ss;
      ss << PrefetchRand();
      ss >> file_id;
      std::ofstream inf((string("dump/") + file_id +
          string("_info.txt")).c_str(), std::ofstream::out);
      inf << image.first << std::endl
          << window[WindowPairDataLayer<Dtype>::X1]+1 << std::endl
          << window[WindowPairDataLayer<Dtype>::Y1]+1 << std::endl
          << window[WindowPairDataLayer<Dtype>::X2]+1 << std::endl
          << window[WindowPairDataLayer<Dtype>::Y2]+1 << std::endl
          << do_mirror << std::endl
          << top_label[item_id] << std::endl
          << is_fg << std::endl;
      inf.close();
      std::ofstream top_data_file((string("dump/") + file_id +
          string("_data.txt")).c_str(),
          std::ofstream::out | std::ofstream::binary);
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            top_data_file.write(reinterpret_cast<char*>(
                &top_data[((item_id * channels + c) * crop_size + h)
                          * crop_size + w]),
                sizeof(Dtype));
          }
        }
      }
      top_data_file.close();
      #endif

      item_id++;
    }
  }
}

INSTANTIATE_CLASS(WindowPairDataLayer);

}  // namespace caffe
