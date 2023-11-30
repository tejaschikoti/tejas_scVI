from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, kl_divergence as kl

# from scvi.models.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial
from scvi.models.log_likelihood import log_mixture_nb
from scvi.models.utils import one_hot
import numpy as np
from scvi.models.modules import DecoderproteinVI, EncoderProteinVI

torch.backends.cudnn.benchmark = True


class ProteinVI(nn.Module):
    def __init__(
        self,
        n_input_proteins: int,
        n_output_proteins: int,
        shared_features: list,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate_decoder: float = 0.2,
        dropout_rate_encoder: float = 0.2,
        # gene_dispersion: str = "gene",
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        reconstruction_loss_gene: str = "nb",
        latent_distribution: str = "ln",
        protein_batch_mask: List[np.ndarray] = None,
        encoder_batch: bool = True,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss_gene = reconstruction_loss_gene
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.shared_features = shared_features
        self.n_input_proteins = n_input_proteins
        self.n_output_proteins = n_output_proteins
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.protein_batch_mask = protein_batch_mask

        # parameters for prior on rate_back (background protein mean)
        if n_batch > 0:
            self.background_pro_alpha = torch.nn.Parameter(
                torch.randn(n_output_proteins, n_batch)
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.clamp(torch.randn(n_output_proteins, n_batch), -10, 1)
            )
        else:
            self.background_pro_alpha = torch.nn.Parameter(
                torch.randn(n_output_proteins)
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.clamp(torch.randn(n_output_proteins), -10, 1)
            )

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(torch.ones(self.n_output_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(torch.ones(self.n_output_proteins, n_batch))
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(torch.ones(self.n_output_proteins, n_labels))
        else:  # protein-cell
            pass

        self.encoder = EncoderProteinVI(
            self.n_input_proteins,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )
        
        self.decoder = DecoderproteinVI(
            n_latent,
            self.n_output_proteins,
            n_layers=n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

    def sample_from_posterior_z(
        self,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        give_mean: bool = False,
        n_samples: int = 5000,
    ) -> torch.Tensor:
        """ Access the tensor of latent values from the posterior

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param batch_index: tensor of batch indices
        :param give_mean: Whether to sample, or give mean of distribution
        :return: tensor of shape ``(batch_size, n_latent)``
        """
        if self.log_variational:
            y = torch.log(1.0 + y)
        qz_m, qz_v, latent, _ = self.encoder(
            y, batch_index
        )
        z = latent["z"]
        if give_mean:
            if self.latent_distribution == "ln":
                samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
                z = self.encoder.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m
        return z

    def get_reconstruction_loss(
        self,
        # x: torch.Tensor,
        y: torch.Tensor,
        # px_: Dict[str, torch.Tensor],
        py_: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        ## Pick shared features
        y_ = y[:,self.shared_features]
        reconst_loss_protein_full = -log_mixture_nb(
            y_, py_["rate_back"], py_["rate_fore"], py_["r"], None, py_["mixing"]
        )
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_protein


    def inference(
        self,
        # x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        transform_batch: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        
    
        y_ = y
        if self.log_variational:
            y_ = torch.log(1.0 + y_)

        qz_m, qz_v, latent, untran_latent = self.encoder(
            y_,None
        )
        z = latent["z"]
        # library_gene = latent["l"]
        untran_z = untran_latent["z"]
        # untran_l = untran_latent["l"]

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.encoder.z_transformation(untran_z)
            # ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            # ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            # untran_l = Normal(ql_m, ql_v.sqrt()).sample()
            # library_gene = self.encoder.l_transformation(untran_l)


        if self.protein_dispersion == "protein-label":
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)

        if self.n_batch > 0:
            py_back_alpha_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_beta_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_alpha_prior = self.background_pro_alpha
            py_back_beta_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(py_back_alpha_prior, py_back_beta_prior)

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        py_, log_pro_back_mean = self.decoder(z,None)
        # px_["r"] = px_r
        py_["r"] = py_r

        return dict(
            py_=py_,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z,
            # ql_m=ql_m,
            # ql_v=ql_v,
            # library_gene=library_gene,
            # untran_l=untran_l,
            log_pro_back_mean=log_pro_back_mean,
        )

    def forward(
        self,
        # x: torch.Tensor,
        y: torch.Tensor,
        # local_l_mean_gene: torch.Tensor,
        # local_l_var_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """ Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param local_l_mean_gene: tensor of means of the prior distribution of latent variable l
         with shape ``(batch_size, 1)````
        :param local_l_var_gene: tensor of variancess of the prior distribution of latent variable l
         with shape ``(batch_size, 1)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        """
        outputs = self.inference(y, batch_index, label)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        z = outputs["z"]
        # ql_m = outputs["ql_m"]
        # ql_v = outputs["ql_v"]
        # px_ = outputs["px_"]
        py_ = outputs["py_"]


        if self.protein_batch_mask is not None:
            pro_batch_mask_minibatch = torch.zeros_like(y)
            for b in np.arange(len(torch.unique(batch_index))):
                b_indices = (batch_index == b).reshape(-1)
                pro_batch_mask_minibatch[b_indices] = torch.tensor(
                    self.protein_batch_mask[b].astype(np.float32), device=y.device
                )
        else:
            pro_batch_mask_minibatch = None

        
        reconst_loss_protein = self.get_reconstruction_loss(
            y, py_, pro_batch_mask_minibatch
        )

        # KL Divergence
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        # kl_div_l_gene = kl(
        #     Normal(ql_m, torch.sqrt(ql_v)),
        #     Normal(local_l_mean_gene, torch.sqrt(local_l_var_gene)),
        # ).sum(dim=1)

        kl_div_back_pro_full = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior
        )
        if pro_batch_mask_minibatch is not None:
            kl_div_back_pro = (pro_batch_mask_minibatch * kl_div_back_pro_full).sum(
                dim=1
            )
        else:
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=1)

        return (
            reconst_loss_protein,
            kl_div_z,
            # kl_div_l_gene,
            kl_div_back_pro,
            z
        )


class ProteinVI_shared(nn.Module):
    def __init__(
        self,
        n_input_proteins: int,
        n_output_proteins: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate_decoder: float = 0.2,
        dropout_rate_encoder: float = 0.2,
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        reconstruction_loss_gene: str = "nb",
        latent_distribution: str = "ln",
        protein_batch_mask: List[np.ndarray] = None,
        encoder_batch: bool = True,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss_gene = reconstruction_loss_gene
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_proteins = n_input_proteins
        self.n_output_proteins = n_output_proteins
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.protein_batch_mask = protein_batch_mask

        # parameters for prior on rate_back (background protein mean)
        if n_batch > 0:
            self.background_pro_alpha = torch.nn.Parameter(
                torch.randn(n_output_proteins, n_batch)
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.clamp(torch.randn(n_output_proteins, n_batch), -10, 1)
            )
        else:
            self.background_pro_alpha = torch.nn.Parameter(
                torch.randn(n_output_proteins)
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.clamp(torch.randn(n_output_proteins), -10, 1)
            )

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(torch.ones(self.n_output_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(torch.ones(self.n_output_proteins, n_batch))
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(torch.ones(self.n_output_proteins, n_labels))
        else:  # protein-cell
            pass

        self.encoder = EncoderProteinVI(
            self.n_input_proteins,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )
        
        self.decoder = DecoderproteinVI(
            n_latent,
            self.n_output_proteins,
            n_layers=n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

    def sample_from_posterior_z(
        self,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        give_mean: bool = False,
        n_samples: int = 5000,
    ) -> torch.Tensor:
        """ Access the tensor of latent values from the posterior

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param batch_index: tensor of batch indices
        :param give_mean: Whether to sample, or give mean of distribution
        :return: tensor of shape ``(batch_size, n_latent)``
        """
        if self.log_variational:
            y = torch.log(1.0 + y)
        qz_m, qz_v, latent, _ = self.encoder(
            y, batch_index
        )
        z = latent["z"]
        if give_mean:
            if self.latent_distribution == "ln":
                samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
                z = self.encoder.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m
        return z

    def get_reconstruction_loss(
        self,
        y: torch.Tensor,
        py_: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        ## Pick shared features
        y_ = y
        reconst_loss_protein_full = -log_mixture_nb(
            y_, py_["rate_back"], py_["rate_fore"], py_["r"], None, py_["mixing"]
        )
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_protein


    def inference(
        self,
        # x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        transform_batch: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        
    
        y_ = y
        if self.log_variational:
            y_ = torch.log(1.0 + y_)

        qz_m, qz_v, latent, untran_latent = self.encoder(
            y_,None
        )
        z = latent["z"]
        untran_z = untran_latent["z"]
        
        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.encoder.z_transformation(untran_z)
            


        if self.protein_dispersion == "protein-label":
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)

        if self.n_batch > 0:
            py_back_alpha_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_beta_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_alpha_prior = self.background_pro_alpha
            py_back_beta_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(py_back_alpha_prior, py_back_beta_prior)

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        py_, log_pro_back_mean = self.decoder(z,None)
        # px_["r"] = px_r
        py_["r"] = py_r

        return dict(
            py_=py_,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z,
            log_pro_back_mean=log_pro_back_mean,
        )

    def forward(
        self,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """ Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param local_l_mean_gene: tensor of means of the prior distribution of latent variable l
         with shape ``(batch_size, 1)````
        :param local_l_var_gene: tensor of variancess of the prior distribution of latent variable l
         with shape ``(batch_size, 1)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        """
        outputs = self.inference(y, batch_index, label)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        z = outputs["z"]
        py_ = outputs["py_"]


        if self.protein_batch_mask is not None:
            pro_batch_mask_minibatch = torch.zeros_like(y)
            for b in np.arange(len(torch.unique(batch_index))):
                b_indices = (batch_index == b).reshape(-1)
                pro_batch_mask_minibatch[b_indices] = torch.tensor(
                    self.protein_batch_mask[b].astype(np.float32), device=y.device
                )
        else:
            pro_batch_mask_minibatch = None

        
        reconst_loss_protein = self.get_reconstruction_loss(
            y, py_, pro_batch_mask_minibatch
        )

        
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        

        kl_div_back_pro_full = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior
        )
        if pro_batch_mask_minibatch is not None:
            kl_div_back_pro = (pro_batch_mask_minibatch * kl_div_back_pro_full).sum(
                dim=1
            )
        else:
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=1)

        return (
            reconst_loss_protein,
            kl_div_z,
            kl_div_back_pro,
            z
        )