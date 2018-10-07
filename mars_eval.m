%% test MARS

feat_path = 'mars_feat.mat';

addpath 'MARS-evaluation/utils/'

track_test = importdata('MARS-evaluation/info/tracks_test_info.mat');
box_feature_test = load(feat_path);
box_feature_test = box_feature_test.features';
video_feat_test = process_box_feat(box_feature_test, track_test); % video video features for test (gallery+query)

query_IDX = importdata('MARS-evaluation/info/query_IDX.mat');  % load pre-defined query index

% train, gallery, and query labels
label_gallery = track_test(:, 3);
label_query = label_gallery(query_IDX);
cam_gallery = track_test(:, 4);
cam_query = cam_gallery(query_IDX);
feat_gallery = video_feat_test;
feat_query = video_feat_test(:, query_IDX);
num_query = size(feat_query, 2);

% re_rank
% dist = re_ranking_wo_query(feat_gallery, feat_query, num_query, 20, 6, 0.3);
% dist = re_ranking_with_query(feat_gallery, feat_query, num_query, 20, 6, 0.3);
% dist = dist(1:num_query, num_query+1:end)';
% original
dist = sqdist(feat_gallery, feat_query);

% Calcuate CMC and mAP
[CMC, map, ~, ~] = evaluation_mars(dist, label_gallery, label_query, cam_gallery, cam_query);
CMC([1,5,10,20])
map
