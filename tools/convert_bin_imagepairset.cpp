// This program converts a set of image pairs to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_bin_imagepairset [FLAGS] datfile catfile infofile

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb", "The backend for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  //if (argc != 4) {
  //  gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_bin_imagepairset");
  //  return 1;
  //}

  bool is_color = !FLAGS_gray;

  //reading dat file demo
  std::ifstream datfile(argv[1], ios::binary|ios::in);
  unsigned int *tmp;
  tmp = new unsigned int;
  datfile.seekg(16);
  datfile.read((char *)tmp, 4);
  unsigned int ht=*tmp;
  datfile.read((char *)tmp, 4);
  unsigned int wd=*tmp;
  std::cout<<"ht:"<<ht<<",wd:"<<wd<<std::endl;
  unsigned int sz=ht*wd;
  delete tmp;
  unsigned char *im1, *im2; 
  im1 = new unsigned char[sz];
  im2 = new unsigned char[sz]; 
  std::cout<<"Reading image 1"<<std::endl;
  datfile.read((char *)im1, sz);
  std::cout<<"Reading image 2"<<std::endl; //useless perturbed image 
  datfile.read((char *)im2, sz);
  std::cout<<"Done"<<std::endl;
  datfile.close();
  delete im1;
  delete im2;

  //reading cat file demo
  std::ifstream catfile(argv[2], ios::binary|ios::in);
  catfile.seekg(20);
  int *label;
  label = new int;
  for(int i=0; i<10; i++){
      catfile.read((char *)label, 4); // reading 4 bits into one integer 
      std::cout<<*label<<std::endl;
  }
  delete label;
  catfile.close();
 
  //reading info file demo
  std::ifstream infofile(argv[3], ios::binary|ios::in);
  infofile.seekg(20);
  int *label1;
  label1 = new int[10];
  for(int i=0; i<10; i++){
      infofile.read((char *)label1, 4); // reading 4 bits into one integer 
      std::cout<<*label1<<std::endl;
  }
  delete label1;
  infofile.close();
 

  //char * tmp;
  //tmp = new char[4];

  //datfile.read(tmp, 4);
  //std::cout<<(int)static_cast<unsigned char>(tmp[0])<<","<<(int)static_cast<unsigned char>(tmp[1])<<","<<(int)static_cast<unsigned char>(tmp[2])<<","<<(int)static_cast<unsigned char>(tmp[3])<<std::endl;
  //datfile.read(tmp, 4);
  //std::cout<<(int)static_cast<unsigned char>(tmp[0])<<","<<(int)static_cast<unsigned char>(tmp[1])<<","<<(int)static_cast<unsigned char>(tmp[2])<<","<<(int)static_cast<unsigned char>(tmp[3])<<std::endl;
  //datfile.read(tmp, 4);
  //std::cout<<(int)static_cast<unsigned char>(tmp[0])<<","<<(int)static_cast<unsigned char>(tmp[1])<<","<<(int)static_cast<unsigned char>(tmp[2])<<","<<(int)static_cast<unsigned char>(tmp[3])<<std::endl;
  //datfile.read(tmp, 4);
  //std::cout<<(int)static_cast<unsigned char>(tmp[0])<<","<<(int)static_cast<unsigned char>(tmp[1])<<","<<(int)static_cast<unsigned char>(tmp[2])<<","<<(int)static_cast<unsigned char>(tmp[3])<<std::endl;
  //datfile.read(tmp, 4);
  //std::cout<<(int)static_cast<unsigned char>(tmp[0])<<","<<(int)static_cast<unsigned char>(tmp[1])<<","<<(int)static_cast<unsigned char>(tmp[2])<<","<<(int)static_cast<unsigned char>(tmp[3])<<std::endl;
  //datfile.read(tmp, 4);
  //std::cout<<(int)static_cast<unsigned char>(tmp[0])<<","<<(int)static_cast<unsigned char>(tmp[1])<<","<<(int)static_cast<unsigned char>(tmp[2])<<","<<(int)static_cast<unsigned char>(tmp[3])<<std::endl;
 

  //std::vector<std::pair<std::pair<string, string>, int> > lines;
  //string filename1, filename2;
  //int label;
  //while (infile >> filename1 >> filename2 >> label) {
  //  lines.push_back(std::make_pair(std::make_pair(filename1, filename2), label));
  //}
  //if (FLAGS_shuffle) {
  //  // randomly shuffle data
  //  LOG(INFO) << "Shuffling data";
  //  shuffle(lines.begin(), lines.end());
  //}
  //LOG(INFO) << "A total of " << lines.size() << " image pairs.";

  //const string& db_backend = FLAGS_backend;
  //const char* db_path = argv[3];

  //int resize_height = std::max<int>(0, FLAGS_resize_height);
  //int resize_width = std::max<int>(0, FLAGS_resize_width);

  //// Open new db
  //// lmdb
  //MDB_env *mdb_env;
  //MDB_dbi mdb_dbi;
  //MDB_val mdb_key, mdb_data;
  //MDB_txn *mdb_txn;
  //// leveldb
  //leveldb::DB* db;
  //leveldb::Options options;
  //options.error_if_exists = true;
  //options.create_if_missing = true;
  //options.write_buffer_size = 268435456;
  //leveldb::WriteBatch* batch = NULL;

  //// Open db
  //if (db_backend == "leveldb") {  // leveldb
  //  LOG(INFO) << "Opening leveldb " << db_path;
  //  leveldb::Status status = leveldb::DB::Open(
  //      options, db_path, &db);
  //  CHECK(status.ok()) << "Failed to open leveldb " << db_path
  //      << ". Is it already existing?";
  //  batch = new leveldb::WriteBatch();
  //} else if (db_backend == "lmdb") {  // lmdb
  //  LOG(INFO) << "Opening lmdb " << db_path;
  //  CHECK_EQ(mkdir(db_path, 0744), 0)
  //      << "mkdir " << db_path << "failed";
  //  CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  //  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
  //      << "mdb_env_set_mapsize failed";
  //  CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
  //      << "mdb_env_open failed";
  //  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
  //      << "mdb_txn_begin failed";
  //  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
  //      << "mdb_open failed. Does the lmdb already exist? ";
  //} else {
  //  LOG(FATAL) << "Unknown db backend " << db_backend;
  //}

  //// Storing to db
  //string root_folder(argv[1]);
  //Datum datum;
  //int count = 0;
  //const int kMaxKeyLength = 256;
  //char key_cstr[kMaxKeyLength];
  //int data_size;
  //bool data_size_initialized = false;

  //for (int line_id = 0; line_id < lines.size(); ++line_id) {
  //  if (!ReadImagePairToDatum(root_folder + lines[line_id].first.first,
  //                            root_folder + lines[line_id].first.second,
  //                            lines[line_id].second, resize_height,
  //                            resize_width, is_color, &datum)) {
  //    continue;
  //  }
  //  if (!data_size_initialized) {
  //    data_size = datum.channels() * datum.height() * datum.width();
  //    data_size_initialized = true;
  //  } else {
  //    const string& data = datum.data();
  //    CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
  //        << data.size();
  //  }
  //  // sequential
  //  snprintf(key_cstr, kMaxKeyLength, "%08d_%s_%s", line_id,
  //      lines[line_id].first.first.c_str(), 
  //      lines[line_id].first.second.c_str());
  //  string value;
  //  datum.SerializeToString(&value);
  //  string keystr(key_cstr);

  //  // Put in db
  //  if (db_backend == "leveldb") {  // leveldb
  //    batch->Put(keystr, value);
  //  } else if (db_backend == "lmdb") {  // lmdb
  //    mdb_data.mv_size = value.size();
  //    mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
  //    mdb_key.mv_size = keystr.size();
  //    mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
  //    CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
  //        << "mdb_put failed";
  //  } else {
  //    LOG(FATAL) << "Unknown db backend " << db_backend;
  //  }

  //  if (++count % 1000 == 0) {
  //    // Commit txn
  //    if (db_backend == "leveldb") {  // leveldb
  //      db->Write(leveldb::WriteOptions(), batch);
  //      delete batch;
  //      batch = new leveldb::WriteBatch();
  //    } else if (db_backend == "lmdb") {  // lmdb
  //      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
  //          << "mdb_txn_commit failed";
  //      CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
  //          << "mdb_txn_begin failed";
  //    } else {
  //      LOG(FATAL) << "Unknown db backend " << db_backend;
  //    }
  //    LOG(ERROR) << "Processed " << count << " files.";
  //  }
  //}
  //// write the last batch
  //if (count % 1000 != 0) {
  //  if (db_backend == "leveldb") {  // leveldb
  //    db->Write(leveldb::WriteOptions(), batch);
  //    delete batch;
  //    delete db;
  //  } else if (db_backend == "lmdb") {  // lmdb
  //    CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
  //    mdb_close(mdb_env, mdb_dbi);
  //    mdb_env_close(mdb_env);
  //  } else {
  //    LOG(FATAL) << "Unknown db backend " << db_backend;
  //  }
  //  LOG(ERROR) << "Processed " << count << " files.";
  //}
  return 0;
}
