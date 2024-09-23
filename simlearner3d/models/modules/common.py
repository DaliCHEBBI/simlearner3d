import torch.nn.functional as F

def balanced_binary_cross_entropy(pred, gt,nogt, pos_w=2.0, neg_w=1.0):
    masked_nogt=nogt.sub(gt)
    # flatten vectors
    pred = pred.view(-1)
    gt = gt.view(-1)
    masked_nogt=masked_nogt.view(-1)
    # select postive/nevative samples
    pos_ind = gt.nonzero().squeeze(-1)
    neg_ind = masked_nogt.nonzero().squeeze(-1)

    # compute weighted loss
    pos_loss = pos_w*F.binary_cross_entropy(pred[pos_ind], gt[pos_ind], reduction='none')
    neg_loss = neg_w*F.binary_cross_entropy(pred[neg_ind], masked_nogt[neg_ind], reduction='none')
    g_loss=pos_loss + neg_loss
    g_loss=g_loss.div(nogt.count_nonzero()+1e-12)
    return g_loss

def mse(coords, coords_gt, prob_gt):

    # flatten vectors
    coords = coords.view(-1, 2)
    coords_gt = coords_gt.view(-1, 2)
    prob_gt = prob_gt.view(-1)

    # select positive samples
    pos_ind = prob_gt.nonzero().squeeze(-1)
    pos_coords = coords[pos_ind, :]
    pos_coords_gt = coords_gt[pos_ind, :]

    return F.mse_loss(pos_coords, pos_coords_gt)


def generate_pointcloud(CUBE, ply_file):
    """
    Generate a colored ply from  the dense cube
    """
    points = []
    for zz in range(CUBE.size()[0]):
        for yy in range(CUBE.size()[1]):
            for xx in range(CUBE.size()[2]):
                val=CUBE[zz,yy,xx]
                points.append("%f %f %f %f %f %f 0\n"%(xx,yy,zz,val,val,val))
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property float red
property float green
property float blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()