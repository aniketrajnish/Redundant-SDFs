from extract import extract_sdf
from visualize import viz_sdf

def main():
    mesh = 'src/data/bunny_mid.obj'
    pts, sdf = extract_sdf(mesh)
    viz_sdf(pts, sdf, mesh)

if __name__ == '__main__':
    main()