# -*- coding: utf-8 -*-
"""
Manual Bot Comparison Script
Python 3
Mejorado: 2025-10-18
@author: z_tjona

Script mejorado para comparar bots de Quarto con:
- Parametrización por línea de comandos
- Guardado automático de resultados
- Manejo robusto de errores
- Soporte para múltiples tipos de bots

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict
import csv

try:
    from quartopy import play_games
except ImportError:
    print("Error: quartopy no está instalado o no se encuentra en el PYTHONPATH")
    print("Asegúrate de tener configurada la variable QUARTOPY_PATH en .env")
    sys.exit(1)

from bot.CNN_bot import Quarto_bot

# Intentar importar CNN_F_bot (opcional)
try:
    from bot.CNN_F_bot import Quarto_bot as F_bot
    HAS_F_BOT = True
except ImportError:
    HAS_F_BOT = False
    print("Advertencia: CNN_F_bot no disponible. Solo se usará CNN_bot.")


class BotComparator:
    """Clase para gestionar comparaciones entre bots."""
    
    def __init__(self, results_dir: str = "comparison_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_bot(self, model_path: str, bot_type: str = "cnn", 
                 temperature: float = 0.1, deterministic: bool = False) -> Optional[object]:
        """
        Carga un bot desde un archivo de modelo.
        
        Args:
            model_path: Ruta al archivo del modelo
            bot_type: Tipo de bot ('cnn' o 'cnn_f')
            temperature: Temperatura para exploración
            deterministic: Si usar modo determinístico
            
        Returns:
            Instancia del bot o None si falla
        """
        model_file = Path(model_path)
        
        if not model_file.exists():
            print(f"Error: El archivo {model_path} no existe")
            return None
        
        try:
            if bot_type.lower() == "cnn":
                bot = Quarto_bot(
                    model_path=str(model_file),
                    deterministic=deterministic,
                    temperature=temperature
                )
            elif bot_type.lower() == "cnn_f":
                if not HAS_F_BOT:
                    print("Error: CNN_F_bot no está disponible")
                    return None
                bot = F_bot(model_path=str(model_file))
            else:
                print(f"Error: Tipo de bot '{bot_type}' no reconocido")
                return None
            
            print(f"✓ Bot cargado: {model_file.name}")
            return bot
            
        except Exception as e:
            print(f"Error al cargar bot desde {model_path}: {e}")
            return None
    
    def compare_bots(self, bot_a: object, bot_b: object,
                    bot_a_name: str, bot_b_name: str,
                    n_matches: int = 500, verbose: bool = False,
                    save_matches: bool = False, mode_2x2: bool = True) -> Dict:
        """
        Compara dos bots jugando partidas en ambas posiciones.
        
        Args:
            bot_a, bot_b: Instancias de los bots
            bot_a_name, bot_b_name: Nombres descriptivos
            n_matches: Número de partidas por configuración
            verbose: Mostrar detalles de cada partida
            save_matches: Guardar partidas individuales
            mode_2x2: Usar modo 2x2
            
        Returns:
            Diccionario con resultados
        """
        print(f"\n{'='*70}")
        print(f"Comparando: {bot_a_name} vs {bot_b_name}")
        print(f"Partidas por posición: {n_matches}")
        print(f"{'='*70}\n")
        
        # Bot A como jugador 1
        print(f"Ronda 1: {bot_a_name} (P1) vs {bot_b_name} (P2)")
        res_1, win_rate_a_p1 = play_games(
            matches=n_matches,
            player1=bot_a,
            player2=bot_b,
            verbose=verbose,
            save_match=save_matches,
            mode_2x2=mode_2x2
        )
        
        # Bot A como jugador 2
        print(f"Ronda 2: {bot_b_name} (P1) vs {bot_a_name} (P2)")
        res_2, win_rate_a_p2 = play_games(
            matches=n_matches,
            player1=bot_b,
            player2=bot_a,
            verbose=verbose,
            save_match=save_matches,
            mode_2x2=mode_2x2
        )
        
        # Calcular estadísticas
        total_matches = n_matches * 2
        
        # Win rate de bot_a
        wins_a_p1 = win_rate_a_p1.get('player1_wins', 0)
        wins_a_p2 = win_rate_a_p2.get('player2_wins', 0)
        total_wins_a = wins_a_p1 + wins_a_p2
        
        # Win rate de bot_b
        wins_b_p1 = win_rate_a_p2.get('player1_wins', 0)
        wins_b_p2 = win_rate_a_p1.get('player2_wins', 0)
        total_wins_b = wins_b_p1 + wins_b_p2
        
        # Empates
        draws_1 = win_rate_a_p1.get('draws', 0)
        draws_2 = win_rate_a_p2.get('draws', 0)
        total_draws = draws_1 + draws_2
        
        results = {
            'timestamp': self.timestamp,
            'bot_a_name': bot_a_name,
            'bot_b_name': bot_b_name,
            'total_matches': total_matches,
            'matches_per_position': n_matches,
            'bot_a': {
                'total_wins': total_wins_a,
                'wins_as_p1': wins_a_p1,
                'wins_as_p2': wins_a_p2,
                'win_rate': total_wins_a / total_matches * 100,
                'win_rate_p1': wins_a_p1 / n_matches * 100,
                'win_rate_p2': wins_a_p2 / n_matches * 100
            },
            'bot_b': {
                'total_wins': total_wins_b,
                'wins_as_p1': wins_b_p1,
                'wins_as_p2': wins_b_p2,
                'win_rate': total_wins_b / total_matches * 100,
                'win_rate_p1': wins_b_p1 / n_matches * 100,
                'win_rate_p2': wins_b_p2 / n_matches * 100
            },
            'draws': {
                'total': total_draws,
                'rate': total_draws / total_matches * 100
            },
            'raw_results': {
                'round_1': win_rate_a_p1,
                'round_2': win_rate_a_p2
            }
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Imprime resultados de forma legible."""
        print(f"\n{'='*70}")
        print("RESULTADOS DE LA COMPARACIÓN")
        print(f"{'='*70}")
        print(f"\nBot A: {results['bot_a_name']}")
        print(f"Bot B: {results['bot_b_name']}")
        print(f"Total de partidas: {results['total_matches']}")
        print(f"\n{'-'*70}")
        print(f"{'Estadística':<30} {'Bot A':<20} {'Bot B':<20}")
        print(f"{'-'*70}")
        print(f"{'Victorias totales':<30} {results['bot_a']['total_wins']:<20} {results['bot_b']['total_wins']:<20}")
        print(f"{'Win rate general':<30} {results['bot_a']['win_rate']:.2f}%{'':<14} {results['bot_b']['win_rate']:.2f}%")
        print(f"{'Victorias como P1':<30} {results['bot_a']['wins_as_p1']:<20} {results['bot_b']['wins_as_p1']:<20}")
        print(f"{'Win rate como P1':<30} {results['bot_a']['win_rate_p1']:.2f}%{'':<14} {results['bot_b']['win_rate_p1']:.2f}%")
        print(f"{'Victorias como P2':<30} {results['bot_a']['wins_as_p2']:<20} {results['bot_b']['wins_as_p2']:<20}")
        print(f"{'Win rate como P2':<30} {results['bot_a']['win_rate_p2']:.2f}%{'':<14} {results['bot_b']['win_rate_p2']:.2f}%")
        print(f"{'-'*70}")
        print(f"{'Empates':<30} {results['draws']['total']:<20}")
        print(f"{'Tasa de empates':<30} {results['draws']['rate']:.2f}%")
        print(f"{'='*70}\n")
        
        # Determinar ganador
        if results['bot_a']['total_wins'] > results['bot_b']['total_wins']:
            winner = results['bot_a_name']
            margin = results['bot_a']['win_rate'] - results['bot_b']['win_rate']
        elif results['bot_b']['total_wins'] > results['bot_a']['total_wins']:
            winner = results['bot_b_name']
            margin = results['bot_b']['win_rate'] - results['bot_a']['win_rate']
        else:
            winner = "Empate"
            margin = 0
        
        print(f"🏆 Ganador: {winner}")
        if winner != "Empate":
            print(f"   Margen: {margin:.2f}%")
        print()
    
    def save_results(self, results: Dict, format: str = "json"):
        """
        Guarda resultados en archivo.
        
        Args:
            results: Diccionario con resultados
            format: Formato de salida ('json' o 'csv')
        """
        base_name = f"comparison_{results['bot_a_name']}_vs_{results['bot_b_name']}_{self.timestamp}"
        
        if format == "json":
            output_file = self.results_dir / f"{base_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✓ Resultados guardados en: {output_file}")
        
        elif format == "csv":
            output_file = self.results_dir / f"{base_name}.csv"
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Métrica', 'Bot A', 'Bot B'])
                writer.writerow(['Nombre', results['bot_a_name'], results['bot_b_name']])
                writer.writerow(['Total partidas', results['total_matches'], ''])
                writer.writerow(['Victorias totales', results['bot_a']['total_wins'], results['bot_b']['total_wins']])
                writer.writerow(['Win rate general (%)', f"{results['bot_a']['win_rate']:.2f}", f"{results['bot_b']['win_rate']:.2f}"])
                writer.writerow(['Victorias como P1', results['bot_a']['wins_as_p1'], results['bot_b']['wins_as_p1']])
                writer.writerow(['Win rate P1 (%)', f"{results['bot_a']['win_rate_p1']:.2f}", f"{results['bot_b']['win_rate_p1']:.2f}"])
                writer.writerow(['Victorias como P2', results['bot_a']['wins_as_p2'], results['bot_b']['wins_as_p2']])
                writer.writerow(['Win rate P2 (%)', f"{results['bot_a']['win_rate_p2']:.2f}", f"{results['bot_b']['win_rate_p2']:.2f}"])
                writer.writerow(['Empates', results['draws']['total'], ''])
                writer.writerow(['Tasa empates (%)', f"{results['draws']['rate']:.2f}", ''])
            print(f"✓ Resultados guardados en: {output_file}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Comparar dos bots de Quarto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s model1.pt model2.pt
  %(prog)s model1.pt model2.pt --matches 1000 --temp 0.05
  %(prog)s model1.pt model2.pt --bot-a-type cnn_f --save-format csv
  %(prog)s model1.pt model2.pt --names "Bot Bueno" "Bot Malo"
        """
    )
    
    parser.add_argument('model_a', help='Ruta al modelo del Bot A')
    parser.add_argument('model_b', help='Ruta al modelo del Bot B')
    
    parser.add_argument('--names', nargs=2, metavar=('NAME_A', 'NAME_B'),
                       help='Nombres descriptivos para los bots')
    
    parser.add_argument('--bot-a-type', choices=['cnn', 'cnn_f'], default='cnn',
                       help='Tipo de bot A (default: cnn)')
    parser.add_argument('--bot-b-type', choices=['cnn', 'cnn_f'], default='cnn',
                       help='Tipo de bot B (default: cnn)')
    
    parser.add_argument('--matches', type=int, default=500,
                       help='Número de partidas por posición (default: 500)')
    parser.add_argument('--temp', type=float, default=0.1,
                       help='Temperatura para exploración (default: 0.1)')
    
    parser.add_argument('--deterministic', action='store_true',
                       help='Usar modo determinístico')
    parser.add_argument('--verbose', action='store_true',
                       help='Mostrar detalles de cada partida')
    parser.add_argument('--save-matches', action='store_true',
                       help='Guardar partidas individuales')
    parser.add_argument('--no-mode-2x2', action='store_true',
                       help='No usar modo 2x2')
    
    parser.add_argument('--save-format', choices=['json', 'csv', 'both'], default='json',
                       help='Formato para guardar resultados (default: json)')
    parser.add_argument('--results-dir', default='comparison_results',
                       help='Directorio para guardar resultados (default: comparison_results)')
    
    args = parser.parse_args()
    
    # Crear comparador
    comparator = BotComparator(results_dir=args.results_dir)
    
    # Determinar nombres
    if args.names:
        bot_a_name, bot_b_name = args.names
    else:
        bot_a_name = Path(args.model_a).stem
        bot_b_name = Path(args.model_b).stem
    
    # Cargar bots
    print("Cargando bots...")
    bot_a = comparator.load_bot(
        args.model_a,
        bot_type=args.bot_a_type,
        temperature=args.temp,
        deterministic=args.deterministic
    )
    
    bot_b = comparator.load_bot(
        args.model_b,
        bot_type=args.bot_b_type,
        temperature=args.temp,
        deterministic=args.deterministic
    )
    
    if bot_a is None or bot_b is None:
        print("Error: No se pudieron cargar los bots")
        sys.exit(1)
    
    # Comparar bots
    results = comparator.compare_bots(
        bot_a, bot_b,
        bot_a_name, bot_b_name,
        n_matches=args.matches,
        verbose=args.verbose,
        save_matches=args.save_matches,
        mode_2x2=not args.no_mode_2x2
    )
    
    # Mostrar resultados
    comparator.print_results(results)
    
    # Guardar resultados
    if args.save_format in ['json', 'both']:
        comparator.save_results(results, format='json')
    if args.save_format in ['csv', 'both']:
        comparator.save_results(results, format='csv')
    
    print("✓ Comparación completada")


if __name__ == "__main__":
    main()