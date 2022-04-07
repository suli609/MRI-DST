from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file',default='./configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',default='./work_dirs/mask_rcnn-based-model.pth')
    parser.add_argument('--out_file', help='out_file name',default = './result/1.jpg')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None  
    model.show_result(args.img, result, out_file=args.out_file,score_thr=args.score_thr)
if __name__ == '__main__':
    main()