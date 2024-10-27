@name recommend_filepath

@desc 推荐一个文件路径

@import FilesysToolKit

1. 获取目录{dir}下的树形结构 -> fileTree
2. 根据{fileTree}，我现在有个文件名为《{filename}》，请帮我推荐合适的目录来存放它，推荐的目录可以包含尚未存在的子目录，请给出不超过 3 个推荐，每行一个。
	请直接输出目录路径，以{dir}开头，不包含文件名 :