import os
import shutil
import streamlit as st

def show_images(images):
    st.image(images, width=30)

def show_first_10_symbols(folder_path):
    image_files = os.listdir(folder_path)[:10]  
    images = [os.path.join(folder_path, img_name) for img_name in image_files]
    show_images(images)

def show_images_in_folder(folder_path):
    image_files = os.listdir(folder_path)
    images = [os.path.join(folder_path, img_name) for img_name in image_files]
    show_images(images)

def main():
    st.title("Cluster Exploration Tool")
    
    dataset_folder = "./dataset"
    all_folders = [folder for folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, folder))]
    
    for i in range(0, len(all_folders), 8):
        clusters_to_show = all_folders[i:i + 8]
        row = st.columns(8)
        for idx, selected_cluster in enumerate(clusters_to_show):
            with row[idx]:
                st.write(f"Cluster: {selected_cluster}")
                cluster_path = os.path.join(dataset_folder, selected_cluster)
                show_first_10_symbols(cluster_path)

    st.title("Cluster Comparison Tool")
    merge_clusters = st.button("Merge Clusters - second into first")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_cluster1 = st.selectbox("Select Cluster 1", all_folders)
        rename_cluster1 = st.text_input("Rename Cluster 1")
        delete_cluster1 = st.button("Delete Cluster 1")
    with col2:
        selected_cluster2 = st.selectbox("Select Cluster 2", all_folders)
        rename_cluster2 = st.text_input("Rename Cluster 2")
        delete_cluster2 = st.button("Delete Cluster 2")
    
    if selected_cluster1 and selected_cluster2:
        cluster1_path = os.path.join(dataset_folder, selected_cluster1)
        cluster2_path = os.path.join(dataset_folder, selected_cluster2)

        if rename_cluster1:
            new_name = os.path.join(dataset_folder, rename_cluster1)
            os.rename(cluster1_path, new_name)
            st.success(f"Cluster '{selected_cluster1}' renamed to '{rename_cluster1}'")
            selected_cluster1 = rename_cluster1
        
        if rename_cluster2:
            new_name = os.path.join(dataset_folder, rename_cluster2)
            os.rename(cluster2_path, new_name)
            st.success(f"Cluster '{selected_cluster2}' renamed to '{rename_cluster2}'")
            selected_cluster2 = rename_cluster2
        
        if delete_cluster1:
            shutil.rmtree(cluster1_path)
            st.warning(f"Deleted Cluster '{selected_cluster1}'")
        
        if delete_cluster2:
            shutil.rmtree(cluster2_path)
            st.warning(f"Deleted Cluster '{selected_cluster2}'")
        
        if merge_clusters:
            for file_name in os.listdir(cluster2_path):
                source_file = os.path.join(cluster2_path, file_name)
                destination_file = os.path.join(cluster1_path, file_name)
                shutil.move(source_file, destination_file)
            
            st.success(f"Cluster '{selected_cluster2}' merged into '{selected_cluster1}'")

        
        with col1:
            st.write("### Cluster 1")
            show_images_in_folder(cluster1_path)
        with col2:
            st.write("### Cluster 2")
            show_images_in_folder(cluster2_path)
      

if __name__ == "__main__":
    main()
